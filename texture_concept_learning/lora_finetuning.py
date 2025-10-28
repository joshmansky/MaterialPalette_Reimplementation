"""
Re-implementation of the LoRA DreamBooth finetuning script.

This version is fully self-contained, removing all CLI argument parsing.
It imports the re-implemented, class-based modules for data, model setup,
and logging.

The main entry point is the `invert()` function, which runs the
training process using a set of hardcoded default configurations.
"""

import os
import math
import itertools
from pathlib import Path
from argparse import Namespace
from typing import Optional

import torch
from tqdm.auto import tqdm
import torch.utils.checkpoint
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils import check_min_version
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# Import our re-implemented modules
from .data_module import DreamBoothDataModule
from .model_setup import LoraModelManager
from .logging_utils import setup_logging

# Will throw error if the minimal version of diffusers is not installed.
check_min_version("0.10.0.dev0")

# --- Hardcoded Default Configuration ---
# All values are taken from the original args.py
DEFAULT_CONFIG = Namespace(
    # Model & Paths
    pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
    revision=None,
    tokenizer_name=None,

    # Seed
    seed=1,
    
    # Dataset
    # data_dir, prompt are provided by invert()
    resolution=512, # Changed from 256 to 512, which is standard
    dataloader_num_workers=1, # My addition for the new data_module

    # LoRA Options
    use_lora=True,
    lora_r=16,
    lora_alpha=17,
    lora_dropout=0.0,
    lora_bias='none',
    lora_text_encoder_r=16,
    lora_text_encoder_alpha=17,
    lora_text_encoder_dropout=0.0,
    lora_text_encoder_bias='none',

    # Training Hyperparameters
    train_text_encoder=True,
    train_batch_size=1,
    max_train_steps=800,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1, # Added this, as it's used in scheduler
    scale_lr=False,
    lr_scheduler='constant',
    lr_warmup_steps=0,
    lr_num_cycles=1,
    lr_power=1.0,
    learning_rate=1e-4,

    # Optimizer
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,

    # Logs & Checkpoints
    checkpointing_steps=800,
    resume_from_checkpoint=None,
    logging_dir='logs',
    report_to='tensorboard', # 'tensorboard' or 'wandb'
    wandb_key=None,
    wandb_project_name=None,
    
    # Validation
    validation_prompt=None,
    num_validation_images=4,
    validation_steps=100,

    # Advanced Options
    use_8bit_adam=False,
    allow_tf32=True,
    mixed_precision='fp16', # 'no', 'fp16', 'bf16'
    enable_xformers_memory_efficient_attention=True,

    # Distributed Training (un-used in this setup but good to have)
    local_rank=-1,
)
# --- End of Configuration ---


def main_training_process(args: Namespace):
    """
    The core training loop, driven by the provided args.
    """
    
    # 1. Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.output_dir / args.logging_dir,
    )
    
    set_seed(args.seed)
    
    # 2. Setup Logging
    logger = setup_logging(args, accelerator)

    # 3. Setup Data
    data_module = DreamBoothDataModule(args)
    # setup() is called inside train_dataloader() if needed
    train_dataloader = data_module.train_dataloader()
    
    # 4. Setup Model, Optimizer, Scheduler
    model_manager = LoraModelManager(args)
    noise_scheduler, text_encoder, vae, unet = model_manager.load_models()
    optimizer = model_manager.create_optimizer(unet, text_encoder)
    lr_scheduler = model_manager.create_scheduler(optimizer)

    # 5. Prepare everything with Accelerator
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler)

    # 6. Set weight types for inference-only models
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # 7. Initialize trackers
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth_lora", config=vars(args))

    # 8. Start Training
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(data_module.train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, dist. & accum.) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # 9. Handle Resuming from Checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = Path(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [d for d in args.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.name.split("-")[1]))
            path = dirs[-1]
        
        if path.exists():
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.name.split("-")[1])
            first_epoch = global_step // len(train_dataloader)
            resume_step = global_step % len(train_dataloader)
        else:
            accelerator.print(f"Checkpoint {path} not found. Starting from scratch.")
            args.resume_from_checkpoint = None # Clear resume path

    # 10. Training Loop
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
    
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, (img_tensor, prompt_ids) in enumerate(train_dataloader):
            # Skip steps if resuming
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # 1. Encode images to latent space
                latents = vae.encode(img_tensor.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2. Sample noise and timesteps
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=latents.device,
                    dtype=torch.long
                )

                # 3. Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 4. Get text embeddings
                encoder_hidden_states = text_encoder(prompt_ids)[0]

                # 5. Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 6. Calculate loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # 7. Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 11. Logging and Checkpointing
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        model_manager.save_weights(
                            accelerator=accelerator,
                            unet=unet,
                            text_encoder=text_encoder if args.train_text_encoder else None,
                            output_dir=args.output_dir,
                            global_step=global_step,
                            save_text_encoder=args.train_text_encoder
                        )

            if global_step >= args.max_train_steps:
                break
    
    # 12. End of Training
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model_manager.save_weights(
            accelerator=accelerator,
            unet=unet,
            text_encoder=text_encoder if args.train_text_encoder else None,
            output_dir=args.output_dir,
            global_step=global_step,
            save_text_encoder=args.train_text_encoder
        )
    
    accelerator.end_training()
    logger.info(f"Training complete. Final weights saved to {args.output_dir / f'checkpoint-{global_step}'}")
    return args.output_dir / f'checkpoint-{global_step}'


# --- Main Functional Interface ---

DEFAULT_PROMPT_TEMPLATE = "a photo of SKS texture"

def invert(
    data_dir: str | Path,
    prompt_token: str,
    output_root_dir: str | Path,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    **kwargs
) -> Path:
    """
    Functional interface for the inversion step.
    
    This is the main entry point for the training process. It uses the
    hardcoded DEFAULT_CONFIG and overrides it with any provided arguments.

    Args:
        data_dir: Path to the directory of cropped training images.
        prompt_token: The unique token for this concept (e.g., "z9k_rock").
        output_root_dir: The root "weights" directory.
                         Final path will be: <output_root_dir>/<data_dir_name>/<prompt_hash>/
        prompt_template: The prompt template. "SKS" will be replaced by the token.
        **kwargs: Any other training parameters to override from DEFAULT_CONFIG.

    Returns:
        The Path to the final checkpoint directory.
    """
    
    # 1. Create a fresh copy of the defaults
    args = Namespace(**vars(DEFAULT_CONFIG))

    # 2. Apply function arguments
    args.data_dir = str(data_dir)
    args.prompt = prompt_template.replace("SKS", prompt_token)
    
    # 3. Apply any other keyword arguments
    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: Ignoring unknown argument '{key}'")

    # 4. Define the final output directory
    data_dir_name = Path(args.data_dir).name
    prompt_hash = args.prompt.replace(' ', '_')
    args.output_dir = Path(output_root_dir) / data_dir_name / prompt_hash
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    # 5. Check if training is already complete
    final_ckpt_dir = args.output_dir / f'checkpoint-{args.max_train_steps}'
    if final_ckpt_dir.exists():
        print(f"Training already complete for this concept. Skipping.")
        print(f"Checkpoint found at: {final_ckpt_dir}")
        return final_ckpt_dir

    # 6. Run the main training process
    print(f"Starting LoRA finetuning for concept: {prompt_token}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Prompt: '{args.prompt}'")
    
    return main_training_process(args)
