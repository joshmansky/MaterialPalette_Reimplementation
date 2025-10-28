"""
tileable_texture_creation.py (Fixed)

Re-implementation of the high-resolution, tileable texture inference script.

This version encapsulates the core tiling logic within a `TiledDiffusionHelper`
class and modularizes the pipeline loading, denoising loop, and image
stitching into distinct functions for clarity and maintainability.

It now correctly exposes the `generate_texture_from_lora` function for import.
"""

import os
import argparse
import random
from pathlib import Path
from itertools import product
from argparse import Namespace
from typing import Tuple, List, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import numpy as np
# import torchvision.transforms.functional as tf # Not used directly, can remove
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from peft import PeftModel, LoraConfig
from PIL import Image

# --- Core Tiling Logic ---

class TiledDiffusionHelper:
    """
    Encapsulates the logic for patch-based, rolling tiled diffusion.
    """
    def __init__(self, tile_factor_k: int):
        self.k = tile_factor_k
        self._roll_x = 0
        self._roll_y = 0

    def _patch_to_grid(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x_ = x.view(n // (self.k**2), self.k**2, c*h*w).transpose(1, -1)
        folded = F.fold(
            x_, output_size=(h * self.k, w * self.k),
            kernel_size=(h, w), stride=(h, w)
        )
        return folded

    def _unpatch_from_grid(self, x: Tensor, padding: int = 0) -> Tensor:
        n, c, kh, kw = x.shape
        h = (kh - 2 * padding) // self.k
        w = (kw - 2 * padding) // self.k
        x_ = F.unfold(
            x, kernel_size=(h + 2 * padding, w + 2 * padding),
            stride=(h, w)
        )
        unfolded = x_.transpose(1, 2).reshape(
            -1, c, h + 2 * padding, w + 2 * padding
        )
        return unfolded

    def pre_step(self, latents: Tensor) -> Tensor:
        self._roll_x = random.randint(0, latents.size(-2))
        self._roll_y = random.randint(0, latents.size(-1))
        latent_grid = self._patch_to_grid(latents)
        latent_grid_rolled = torch.roll(
            latent_grid, (self._roll_x, self._roll_y), dims=(2, 3)
        )
        return self._unpatch_from_grid(latent_grid_rolled)

    def post_step(self, noise_pred: Tensor) -> Tensor:
        noise_grid = self._patch_to_grid(noise_pred)
        noise_grid_unrolled = torch.roll(
            noise_grid, (-self._roll_x, -self._roll_y), dims=(2, 3)
        )
        return self._unpatch_from_grid(noise_grid_unrolled)

# --- Pipeline Setup ---

def load_lora_pipeline( # Renamed from load_pipeline for clarity
    lora_checkpoint_path: Path,
    device: torch.device
) -> StableDiffusionPipeline:
    """Loads a Stable Diffusion pipeline with PEFT LoRA weights."""

    print(f'Loading LoRA from {lora_checkpoint_path}...')
    unet_dir = lora_checkpoint_path / "unet"
    text_encoder_dir = lora_checkpoint_path / "text_encoder"

    # Find base model from config
    config_path = text_encoder_dir / "adapter_config.json"
    if not config_path.exists():
        # Try unet config if text encoder wasn't trained/saved
        config_path_unet = unet_dir / "adapter_config.json"
        if config_path_unet.exists():
             lora_config = LoraConfig.from_pretrained(unet_dir)
             print("Warning: Text encoder LoRA config not found, reading base model from UNet config.")
        else:
            raise FileNotFoundError(f"Adapter config not found at {config_path} or {config_path_unet}")
    else:
         lora_config = LoraConfig.from_pretrained(text_encoder_dir)

    base_model_path = lora_config.base_model_name_or_path
    if not base_model_path:
         raise ValueError("Base model path not found in LoRA config.")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        local_files_only=True, # Assume base model is cached locally
        safety_checker=None,
    )

    # Load PEFT models
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet, str(unet_dir), adapter_name="default"
    )
    if text_encoder_dir.exists():
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, str(text_encoder_dir), adapter_name="default"
        )
        print("Loaded LoRA weights into Text Encoder.")
    else:
        print("Warning: Text encoder LoRA weights not found. Using base text encoder.")


    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Apply half precision
    if hasattr(pipe, "unet") and hasattr(pipe.unet, "half"):
        pipe.unet.half()
    if hasattr(pipe, "text_encoder") and hasattr(pipe.text_encoder, "half"):
        pipe.text_encoder.half()

    return pipe

# --- Denoising and Stitching ---

def run_denoising_loop(
    pipe: StableDiffusionPipeline,
    tiler: TiledDiffusionHelper,
    prompt_embeds: Tensor,
    args: Namespace,
    device: torch.device
) -> Tensor:
    """Runs the main diffusion loop with tiling logic."""
    k = tiler.k
    batch_size_tiles = k * k
    guidance_scale = 7.5

    generator = torch.Generator(device).manual_seed(args.seed)
    latents = pipe.prepare_latents(
        batch_size_tiles,
        pipe.unet.config.in_channels,
        512, 512, # Base tile size
        prompt_embeds.dtype,
        device,
        generator,
    )
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)

    for t in tqdm(timesteps, desc="Denoising"):
        latent_model_input = torch.cat([latents] * 2) # For CFG
        latent_model_input = pipe.scheduler.scale_model_input(
            latent_model_input, t
        )
        latent_model_input_tiled = tiler.pre_step(latent_model_input)

        # Predict noise (chunked for VRAM)
        noise_pred_chunks = []
        # Calculate chunk size dynamically based on batch size and a heuristic (e.g., max 16 per chunk)
        num_chunks = max(1, (len(latent_model_input_tiled) + 15) // 16) # Ensure at least 1 chunk
        for latent_chunk, prompt_chunk in zip(
            latent_model_input_tiled.chunk(num_chunks),
            prompt_embeds.chunk(num_chunks)
        ):
            pred = pipe.unet(
                latent_chunk, t, encoder_hidden_states=prompt_chunk
            ).sample
            noise_pred_chunks.append(pred)

        noise_pred_tiled = torch.cat(noise_pred_chunks)
        noise_pred = tiler.post_step(noise_pred_tiled)

        # CFG calculation
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Scheduler step
        latents = pipe.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs, return_dict=False
        )[0]

    return latents


def _create_blending_kernel(
    full_tile_size: int,
    overlap_px: int,
    device: torch.device
) -> Tensor:
    """Creates a 2D linear 'tent' blending kernel."""
    ramp = torch.linspace(0, 1, overlap_px, device=device)
    core_size = full_tile_size - 2 * overlap_px
    kernel_1d = torch.cat(
        [ramp, torch.ones(core_size, device=device), ramp.flip(0)]
    )
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d[None, None, :, :]


def stitch_output_image(
    pipe: StableDiffusionPipeline,
    final_latents: Tensor,
    tiler: TiledDiffusionHelper,
    args: Namespace,
    device: torch.device
) -> Image.Image:
    """Decodes and stitches the final latents into a single image."""
    k = tiler.k

    if args.stitch_mode == 'concat':
        decoded_tiles = pipe.vae.decode(
            final_latents / pipe.vae.config.scaling_factor
        ).sample
        img_grid = pipe.image_processor.postprocess(
            decoded_tiles, output_type='pt', do_denormalize=[True] * len(decoded_tiles)
        )
        # Save directly, no need for temp file load
        save_image(img_grid, args.output_file, nrow=k, padding=0)
        return Image.open(args.output_file).convert("RGB") # Reopen to return PIL

    # --- Logic for 'mean' or 'wmean' ---
    latent_pad_px = 1
    pixel_pad_px = latent_pad_px * pipe.vae_scale_factor
    total_overlap_px = 2 * pixel_pad_px
    base_tile_size = 512

    latent_grid = tiler._patch_to_grid(final_latents)
    latent_grid_padded = F.pad(latent_grid, (latent_pad_px,) * 4, mode='circular')
    overlapping_latents = tiler._unpatch_from_grid(latent_grid_padded, padding=latent_pad_px)

    # Decode overlapping tiles (chunked for VRAM)
    decoded_tiles = []
    num_chunks = max(1, (len(overlapping_latents) + 15) // 16)
    for chunk in overlapping_latents.chunk(num_chunks):
        decoded = pipe.vae.decode(chunk / pipe.vae.config.scaling_factor).sample
        decoded_tiles.append(decoded)
    decoded_tiles = torch.cat(decoded_tiles).to(device)

    # Apply blending
    full_tile_size_px = base_tile_size + total_overlap_px
    if args.stitch_mode == 'mean':
        blended_tiles = decoded_tiles
        blended_tiles[:, :, :total_overlap_px, :] /= 2.
        blended_tiles[:, :, -total_overlap_px:, :] /= 2.
        blended_tiles[:, :, :, :total_overlap_px] /= 2.
        blended_tiles[:, :, :, -total_overlap_px:] /= 2.
    elif args.stitch_mode == 'wmean':
        kernel = _create_blending_kernel(full_tile_size_px, pixel_pad_px, device)
        blended_tiles = decoded_tiles * kernel
    else: # Should not happen with choices in argparser
        raise ValueError(f"Unknown stitch mode: {args.stitch_mode}")

    # Stitch blended tiles onto canvas
    final_canvas_size = base_tile_size * k
    canvas = torch.zeros(
        1, 3, final_canvas_size + total_overlap_px, final_canvas_size + total_overlap_px,
        device=device, dtype=blended_tiles.dtype # Match dtype
    )
    idx = 0
    for r, c in product(range(k), range(k)):
        x_start, y_start = c * base_tile_size, r * base_tile_size
        canvas[:, :, y_start : y_start + full_tile_size_px, x_start : x_start + full_tile_size_px] += blended_tiles[idx]
        idx += 1

    # Fold circular padding
    canvas[:, :, -total_overlap_px:-pixel_pad_px, :] += canvas[:, :, :pixel_pad_px, :]
    canvas[:, :, pixel_pad_px:total_overlap_px, :] += canvas[:, :, -pixel_pad_px:, :]
    canvas[:, :, :, -total_overlap_px:-pixel_pad_px] += canvas[:, :, :, :pixel_pad_px]
    canvas[:, :, :, pixel_pad_px:total_overlap_px] += canvas[:, :, :, -pixel_pad_px:]

    # Trim padding
    final_image_tensor = canvas[:, :, pixel_pad_px:-pixel_pad_px, pixel_pad_px:-pixel_pad_px]

    # Convert to PIL
    return pipe.image_processor.postprocess(
        final_image_tensor, output_type='pil', do_denormalize=[True]
    )[0]


# --- Main Functional Interface ---

PROMPT_MAP = {
    'p1': 'top view realistic texture of SKS',
    'p2': 'top view realistic SKS texture',
    'p3': 'high resolution realistic SKS texture in top view',
    'p4': 'realistic SKS texture in top view',
}

DEFAULT_GENERATION_CONFIG = Namespace(
    stitch_mode='wmean',
    resolution=1024,
    prompt_key='p1',
    seed=42,
    # renorm=False, # Renorm handled separately
    num_inference_steps=50,
)

@torch.no_grad()
def generate_texture_from_lora(
    lora_path: str | Path,
    output_dir: str | Path,
    prompt_token: str,
    **kwargs
) -> Path:
    """
    Generates a seamless, tileable texture using trained LoRA weights.

    Args:
        lora_path: Path to the LoRA checkpoint directory (e.g., .../checkpoint-800).
        output_dir: Directory to save the generated texture.
        prompt_token: The unique token used during LoRA training (e.g., "example_sks_texture").
        **kwargs: Overrides for DEFAULT_GENERATION_CONFIG (resolution, seed, etc.).

    Returns:
        The Path to the saved texture image file.
    """

    # 1. Setup Configuration
    args = Namespace(**vars(DEFAULT_GENERATION_CONFIG))
    args.lora_path = Path(lora_path) # Changed 'path' to 'lora_path' for clarity
    args.output_dir = Path(output_dir)

    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: Ignoring unknown generation argument '{key}'")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Define Output Filename ---
    reso_map = { 512: 'hK', 1024: '1K', 2048: '2K', 4096: '4K', 8192: '8K' }
    prompt_str = PROMPT_MAP[args.prompt_key].replace(' ', '-').replace('SKS', 'o')
    reso_str = reso_map.get(args.resolution, f'{args.resolution}px')

    output_file = args.output_dir / (
        f"{prompt_token}_{reso_str}_"
        f"t{args.num_inference_steps}_{args.stitch_mode}_"
        f"{prompt_str}_{args.seed}.png"
    )
    args.output_file = output_file # Store for stitch function

    if output_file.exists():
        print(f"File already exists, skipping generation: {output_file}")
        return output_file

    # 2. Setup Device and Seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)


    # 3. Load Model with LoRA
    pipe = load_lora_pipeline(args, device)

    # 4. Get Prompt Embeddings
    prompt = PROMPT_MAP[args.prompt_key].replace("SKS", prompt_token)
    negative_prompt = "lowres, error, cropped, worst quality, low quality, artifact, signature, watermark, text, words"
    print(f"Using prompt: '{prompt}'")

    k = args.resolution // 512
    prompt_embeds = pipe._encode_prompt(
        prompt, device, k * k,
        True, # do_classifier_free_guidance
        negative_prompt
    )

    # 5. Initialize Tiler
    tiler = TiledDiffusionHelper(tile_factor_k=k)

    # 6. Run Denoising Loop
    print(f"Running denoising for {output_file.name}...")
    final_latents = run_denoising_loop(pipe, tiler, prompt_embeds, args, device)

    # 7. Stitch Output Image
    print("Denoising complete. Stitching image...")
    final_image = stitch_output_image(pipe, final_latents, tiler, args, device)

    # 8. Save
    final_image.save(output_file)
    print(f"Successfully saved image to {output_file}")

    # Renormalization is now handled externally by the pipeline script

    return output_file


# --- Main Execution Block (for direct script running) ---

def get_cli_args() -> Namespace:
    """Parses command-line arguments specific to this script."""
    parser = argparse.ArgumentParser(
        description="Generate tileable textures using trained LoRA weights."
    )
    parser.add_argument(
        'lora_path', type=Path,
        help="Path to the LoRA checkpoint directory (e.g., .../checkpoint-800)."
    )
    parser.add_argument(
        '--output_dir', type=Path, default=None,
        help="Directory to save the output. (Default: <lora_path>/outputs)"
    )
    parser.add_argument(
        '--prompt_token', type=str, required=True, # Require token if run directly
        help="The unique token used during LoRA training (e.g., 'example_sks_texture')."
    )
    parser.add_argument(
        '--stitch_mode', type=str, default=DEFAULT_GENERATION_CONFIG.stitch_mode,
        choices=['concat', 'mean', 'wmean']
    )
    parser.add_argument(
        '--resolution', default=DEFAULT_GENERATION_CONFIG.resolution, type=int,
        choices=[512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument(
        '--prompt_key', type=str, default=DEFAULT_GENERATION_CONFIG.prompt_key,
        choices=PROMPT_MAP.keys()
    )
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_GENERATION_CONFIG.seed
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=DEFAULT_GENERATION_CONFIG.num_inference_steps
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cli_args = get_cli_args()
    output_directory = cli_args.output_dir or cli_args.lora_path / "outputs"

    generate_texture_from_lora(
        lora_path=cli_args.lora_path,
        output_dir=output_directory,
        prompt_token=cli_args.prompt_token,
        # Pass CLI args as kwargs to override defaults
        stitch_mode=cli_args.stitch_mode,
        resolution=cli_args.resolution,
        prompt_key=cli_args.prompt_key,
        seed=cli_args.seed,
        num_inference_steps=cli_args.num_inference_steps
    )
