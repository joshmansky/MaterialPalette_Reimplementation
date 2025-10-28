"""
infer.py (Re-implemented)

Re-implementation of the high-resolution, tileable texture inference script.

This version encapsulates the core tiling logic within a `TiledDiffusionHelper`
class and modularizes the pipeline loading, denoising loop, and image
stitching into distinct functions for clarity and maintainability.
"""

import os
import argparse
import random
from pathlib import Path
from itertools import product
from argparse import Namespace
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as tf
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from peft import PeftModel, LoraConfig
from PIL import Image

# --- Core Tiling Logic ---

class TiledDiffusionHelper:
    """
    Encapsulates the logic for patch-based, rolling tiled diffusion.
    
    This class handles the patch/unpatch and roll/unroll operations
    that happen before and after each UNet call, enabling the
    generation of seamlessly tileable images.
    """
    
    def __init__(self, tile_factor_k: int):
        """
        Args:
            tile_factor_k: The number of tiles per side
                           (e.g., 2 for a 2x2 grid, 1024x1024 total).
        """
        self.k = tile_factor_k
        self._roll_x = 0
        self._roll_y = 0

    def _patch_to_grid(self, x: Tensor) -> Tensor:
        """Converts a batch of (k*k) tiles into a single grid image."""
        n, c, h, w = x.shape
        # (n*k*k, c, h, w) -> (n, k*k, c*h*w) -> (n, c*h*w, k*k)
        x_ = x.view(n // (self.k**2), self.k**2, c*h*w).transpose(1, -1)
        # (n, c*h*w, k*k) -> (n, c, h*k, w*k)
        folded = F.fold(
            x_, output_size=(h * self.k, w * self.k),
            kernel_size=(h, w), stride=(h, w)
        )
        return folded

    def _unpatch_from_grid(self, x: Tensor, padding: int = 0) -> Tensor:
        """Splits a grid image into a batch of (k*k) tiles."""
        n, c, kh, kw = x.shape
        h = (kh - 2 * padding) // self.k
        w = (kw - 2 * padding) // self.k
        
        # (n, c, h*k+2p, w*k+2p) -> (n, c*(h+2p)*(w+2p), k*k)
        x_ = F.unfold(
            x, kernel_size=(h + 2 * padding, w + 2 * padding),
            stride=(h, w)
        )
        # (n, c*(h+2p)*(w+2p), k*k) -> (n*k*k, c, h+2p, w+2p)
        unfolded = x_.transpose(1, 2).reshape(
            -1, c, h + 2 * padding, w + 2 * padding
        )
        return unfolded

    def pre_step(self, latents: Tensor) -> Tensor:
        """
        To be called *before* the UNet.
        Patches, rolls, and unpatches the latents.
        """
        # 1. Get a new random roll for this step
        self._roll_x = random.randint(0, latents.size(-2))
        self._roll_y = random.randint(0, latents.size(-1))
        
        # 2. Patch (Batch of tiles -> Single grid)
        latent_grid = self._patch_to_grid(latents)
        
        # 3. Roll (Circularly shift the grid)
        latent_grid_rolled = torch.roll(
            latent_grid, (self._roll_x, self._roll_y), dims=(2, 3)
        )
        
        # 4. Unpatch (Rolled grid -> Batch of tiles)
        return self._unpatch_from_grid(latent_grid_rolled)

    def post_step(self, noise_pred: Tensor) -> Tensor:
        """
        To be called *after* the UNet.
        Patches, un-rolls, and unpatches the noise prediction.
        """
        # 1. Patch (Batch of tiles -> Single grid)
        noise_grid = self._patch_to_grid(noise_pred)
        
        # 2. Un-roll (Shift the grid back)
        noise_grid_unrolled = torch.roll(
            noise_grid, (-self._roll_x, -self._roll_y), dims=(2, 3)
        )
        
        # 3. Unpatch (Unrolled grid -> Batch of tiles)
        return self._unpatch_from_grid(noise_grid_unrolled)


# --- Pipeline Setup ---

def load_pipeline(
    args: Namespace, device: torch.device
) -> StableDiffusionPipeline:
    """Loads and configures the SD pipeline (LoRA or vanilla)."""
    
    if args.token is None:
        # Load LoRA pipeline
        print(f'Loading LoRA from {args.path}...')
        unet_dir = args.path / "unet"
        text_encoder_dir = args.path / "text_encoder"
        
        # Find base model from config
        config_path = text_encoder_dir / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Adapter config not found at {config_path}")
            
        lora_config = LoraConfig.from_pretrained(text_encoder_dir)
        base_model_path = lora_config.base_model_name_or_path
        
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
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
        
    else:
        # Load vanilla pipeline
        print(f'Loading vanilla SD 1.5 for token: {args.token}...')
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            revision="fp16",
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None,
        )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
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
    
    # 1. Setup parameters
    k = tiler.k
    batch_size_tiles = k * k
    guidance_scale = 7.5
    
    # 2. Prepare latents
    generator = torch.Generator(device).manual_seed(args.seed)
    latents = pipe.prepare_latents(
        batch_size_tiles,
        pipe.unet.config.in_channels,
        512, 512,
        prompt_embeds.dtype,
        device,
        generator,
    )

    # 3. Prepare timesteps
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)

    # 4. Denoising loop
    for t in tqdm(timesteps, desc="Denoising"):
        # 1. CFG setup
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(
            latent_model_input, t
        )

        # 2. Tiling Pre-step (Patch, Roll, Unpatch)
        latent_model_input_tiled = tiler.pre_step(latent_model_input)
        
        # 3. UNet prediction (in chunks for VRAM)
        noise_pred_chunks = []
        chunk_size = (len(latent_model_input_tiled) // 16) or 1
        for latent_chunk, prompt_chunk \
            in zip(latent_model_input_tiled.chunk(chunk_size), 
                   prompt_embeds.chunk(chunk_size)):
            
            pred = pipe.unet(
                latent_chunk, t, encoder_hidden_states=prompt_chunk
            ).sample
            noise_pred_chunks.append(pred)
        
        noise_pred_tiled = torch.cat(noise_pred_chunks)

        # 4. Tiling Post-step (Patch, Un-roll, Unpatch)
        noise_pred = tiler.post_step(noise_pred_tiled)

        # 5. CFG calculation
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * \
                     (noise_pred_text - noise_pred_uncond)

        # 6. Scheduler step
        latents = pipe.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs, return_dict=False
        )[0]
        
    return latents


def _create_blending_kernel(
    full_tile_size: int,
    overlap_px: int,
    device: torch.device
) -> Tensor:
    """
    Creates a 2D linear "tent" blending kernel.
    Shape: [1, 1, full_tile_size, full_tile_size]
    """
    # 1D ramp: 0 -> 1 over `overlap_px`
    ramp = torch.linspace(0, 1, overlap_px, device=device)
    
    # 1D kernel: [ramp, ones, flipped_ramp]
    core_size = full_tile_size - 2 * overlap_px
    kernel_1d = torch.cat(
        [ramp, torch.ones(core_size, device=device), ramp.flip(0)]
    )
    
    # 2D kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d[None, None, :, :]


def stitch_output_image(
    pipe: StableDiffusionPipeline,
    final_latents: Tensor,
    tiler: TiledDiffusionHelper,
    args: Namespace,
    device: torch.device
) -> Image.Image:
    """
    Decodes and stitches the final latents into a single image
    using the specified stitch mode.
    """
    k = tiler.k
    
    if args.stitch_mode == 'concat':
        # Decode all tiles and save as a simple grid
        decoded_tiles = pipe.vae.decode(
            final_latents / pipe.vae.config.scaling_factor
        ).sample
        # Rearrange tiles into a grid and save
        img_grid = pipe.image_processor.postprocess(
            decoded_tiles, output_type='pt', do_denormalize=[True] * len(decoded_tiles)
        )
        # Create a temp file to load as PIL (or save directly)
        temp_path = args.output_file.parent / "temp_concat.png"
        save_image(img_grid, temp_path, nrow=k, padding=0)
        return Image.open(temp_path).convert("RGB")
        
    # --- Logic for 'mean' or 'wmean' ---
    
    # 1. Setup padding
    latent_pad_px = 1  # Overlap in latent space
    pixel_pad_px = latent_pad_px * pipe.vae_scale_factor  # 1*8 = 8
    total_overlap_px = 2 * pixel_pad_px                   # 16
    base_tile_size = 512

    # 2. Get overlapping latent tiles
    # (k*k, c, 64, 64) -> (1, c, 64*k, 64*k)
    latent_grid = tiler._patch_to_grid(final_latents)
    # Pad grid circularly
    latent_grid_padded = F.pad(
        latent_grid, (latent_pad_px,) * 4, mode='circular'
    )
    # (1, c, 64*k+2, 64*k+2) -> (k*k, c, 64+2, 64+2)
    overlapping_latents = tiler._unpatch_from_grid(
        latent_grid_padded, padding=latent_pad_px
    )

    # 3. Decode all overlapping tiles
    decoded_tiles = []
    chunk_size = (len(overlapping_latents) // 16) or 1
    for chunk in overlapping_latents.chunk(chunk_size):
        decoded = pipe.vae.decode(
            chunk / pipe.vae.config.scaling_factor
        ).sample
        decoded_tiles.append(decoded)
    decoded_tiles = torch.cat(decoded_tiles).to(device) # [k*k, 3, 512+16, 512+16]

    # 4. Apply blending
    full_tile_size_px = base_tile_size + total_overlap_px
    
    if args.stitch_mode == 'mean':
        # Naive mean: divide overlap regions by 2
        blended_tiles = decoded_tiles
        blended_tiles[:, :, :total_overlap_px, :] /= 2.
        blended_tiles[:, :, -total_overlap_px:, :] /= 2.
        blended_tiles[:, :, :, :total_overlap_px] /= 2.
        blended_tiles[:, :, :, -total_overlap_px:] /= 2.
        
    elif args.stitch_mode == 'wmean':
        # Weighted mean: apply linear falloff kernel
        kernel = _create_blending_kernel(
            full_tile_size_px, pixel_pad_px, device
        )
        blended_tiles = decoded_tiles * kernel

    # 5. Stitch blended tiles onto a final canvas
    final_canvas_size = base_tile_size * k
    canvas = torch.zeros(
        1, 3, final_canvas_size + total_overlap_px,
        final_canvas_size + total_overlap_px,
        device=device
    )

    idx = 0
    for r, c in product(range(k), range(k)):
        x_start, y_start = c * base_tile_size, r * base_tile_size
        canvas[
            :, :, y_start : y_start + full_tile_size_px,
            x_start : x_start + full_tile_size_px
        ] += blended_tiles[idx]
        idx += 1

    # 6. Fold circular padding back in
    # Add top overlap to bottom
    canvas[:, :, -total_overlap_px:-pixel_pad_px, :] += \
        canvas[:, :, :pixel_pad_px, :]
    # Add bottom overlap to top
    canvas[:, :, pixel_pad_px:total_overlap_px, :] += \
        canvas[:, :, -pixel_pad_px:, :]
    # Add left overlap to right
    canvas[:, :, :, -total_overlap_px:-pixel_pad_px] += \
        canvas[:, :, :, :pixel_pad_px]
    # Add right overlap to left
    canvas[:, :, :, pixel_pad_px:total_overlap_px] += \
        canvas[:, :, :, -pixel_pad_px:]

    # 7. Trim padding to get final image
    final_image_tensor = canvas[
        :, :, pixel_pad_px:-pixel_pad_px, pixel_pad_px:-pixel_pad_px
    ]
    
    # 8. Convert to PIL
    return pipe.image_processor.postprocess(
        final_image_tensor, output_type='pil', do_denormalize=[True]
    )[0]


# --- Main Execution ---

def get_args() -> Namespace:
    """Parses and validates command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-implemented inference script for tileable textures."
    )
    # ... (All parser.add_argument calls from the original) ...
    parser.add_argument('path', type=Path, default=None,
                        help="Path to the LoRA checkpoint directory.")
    parser.add_argument('--outdir', type=Path, default=None,
                        help="Optional output directory.")
    parser.add_argument('--token', type=str, default=None,
                        help="Instance token. If set, uses vanilla SD.")
    parser.add_argument('--stitch_mode', type=str, default='wmean',
                        choices=['concat', 'mean', 'wmean'])
    parser.add_argument('--resolution', default=1024, type=int)
    parser.add_argument('--prompt', type=str, default='p1',
                        choices=['p1', 'p2', 'p3', 'p4'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--renorm', action="store_true", default=False)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    
    args = parser.parse_args()
    
    # --- Validate paths and set output file ---
    if args.token is None and not args.path:
        raise ValueError("Must provide --path to LoRA checkpoint if --token is not set.")
        
    if args.path:
        outdir = args.path / 'outputs'
        outdir.mkdir(exist_ok=True)
    elif args.outdir:
        outdir = args.outdir
        outdir.mkdir(exist_ok=True, parents=True)
    else:
        raise ValueError("Must provide either --path or --outdir.")
    
    # Create filename
    prompt_map = {
        'p1': 'top-view-realistic-texture-of-o',
        'p2': 'top-view-realistic-o-texture',
        'p3': 'high-resolution-realistic-o-texture-in-top-view',
        'p4': 'realistic-o-texture-in-top-view',
    }
    reso_map = {
        512: 'hK', 1024: '1K', 2048: '2K', 4096: '4K', 8192: '8K'
    }
    token_str = args.token or "lora"
    
    args.output_file = outdir / (
        f"{token_str}_{reso_map[args.resolution]}_"
        f"t{args.num_inference_steps}_{args.stitch_mode}_"
        f"{prompt_map[args.prompt]}_{args.seed}.png"
    )
    
    return args

def main():
    args = get_args()
    
    if args.output_file.exists():
        print(f"File already exists, skipping: {args.output_file}")
        return

    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    
    # 2. Load Model
    pipe = load_pipeline(args, device)
    
    # 3. Get Prompt
    prompt_templates = dict(
        p1='top view realistic texture of {}',
        p2='top view realistic {} texture',
        p3='high resolution realistic {} texture in top view',
        p4='realistic {} texture in top view',
    )
    token = args.token or "azertyuiop" # Placeholder for LoRA
    prompt = prompt_templates[args.prompt].format(token)
    negative_prompt = "lowres, error, cropped, worst quality, low quality"
    
    # 4. Tiler and Embeddings
    k = args.resolution // 512
    tiler = TiledDiffusionHelper(tile_factor_k=k)
    
    prompt_embeds = pipe._encode_prompt(
        prompt, device, k * k,
        True, # do_classifier_free_guidance
        negative_prompt
    )

    # 5. Run Denoising
    print(f"Running denoising for {args.output_file.name}...")
    final_latents = run_denoising_loop(pipe, tiler, prompt_embeds, args, device)
    
    # 6. Stitch Image
    print("Denoising complete. Stitching image...")
    final_image = stitch_output_image(pipe, final_latents, tiler, args, device)
    
    # 7. Save
    final_image.save(args.output_file)
    print(f"Successfully saved image to {args.output_file}")

    # 8. Renormalization (optional)
    if args.renorm:
        print("Applying renormalization...")
        # This assumes the new `renormalization.py` is in the same dir
        try:
            # We need to find the source/mask paths
            # This logic is still brittle, as in the original
            proj_dir = Path(*args.path.parts[:-6])
            mask_name = args.path.parts[-5]
            
            source_img = next(proj_dir.glob("*.jpg")) # Find first jpg
            mask_file = proj_dir / 'masks' / f'{mask_name}.png'
            
            renorm_out_dir = args.output_file.parent.parent / 'out_renorm'
            renorm_out_dir.mkdir(exist_ok=True)
            renorm_out_path = renorm_out_dir / args.output_file.name
            
            from renormalization import Renormalizer # Import new class
            
            normalizer = Renormalizer(
                source_image_path=source_img,
                mask_path=mask_file,
                device=device
            )
            normalizer.normalize_image(
                target_image_path=args.output_file,
                output_path=renorm_out_path
            )
            print(f"Saved renormalized image to {renorm_out_path}")
            
        except Exception as e:
            print(f"Could not apply renormalization. Error: {e}")
            print("You may need to run it manually with the correct paths.")

if __name__ == "__main__":
    main()