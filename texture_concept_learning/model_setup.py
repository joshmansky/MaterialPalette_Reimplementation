"""
Re-implementation of the model and optimizer setup logic.

This module provides a `LoraModelManager` class that handles:
- Loading and configuring the UNet, VAE, and TextEncoder.
- Applying LoRA to the models.
- Creating the optimizer and learning rate scheduler.
- Saving the final LoRA weights.
"""

import itertools
from pathlib import Path
from argparse import Namespace
from typing import Tuple, Any, Iterable

import torch
import torch.optim
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers import (
    AutoencoderKL, 
    DDPMScheduler, 
    UNet2DConditionModel
)
from transformers import CLIPTextModel


class LoraModelManager:
    """
    Manages the setup of models, optimizer, and scheduler for LoRA training.
    """
    
    def __init__(self, args: Namespace):
        """
        Args:
            args: The global Namespace object with all training settings.
        """
        self.args = args
        self.noise_scheduler: DDPMScheduler
        self.text_encoder: CLIPTextModel
        self.vae: AutoencoderKL
        self.unet: UNet2DConditionModel
    
    def load_models(self) -> Tuple[DDPMScheduler, CLIPTextModel, AutoencoderKL, UNet2DConditionModel]:
        """
        Loads and configures all core diffusion models and applies LoRA.
        """
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000
        )

        # 1. Load Text Encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision
        )

        # 2. Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision
        )
        self.vae.requires_grad_(False) # VAE is never trained

        # 3. Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision
        )

        # 4. Apply LoRA and advanced settings
        self._apply_lora_and_optimizations()

        return self.noise_scheduler, self.text_encoder, self.vae, self.unet

    def _apply_lora_and_optimizations(self):
        """Internal helper to configure LoRA and model optimizations."""
        
        # --- Apply LoRA ---
        if self.args.train_text_encoder:
            if self.args.use_lora:
                print("Applying LoRA to Text Encoder...")
                config = LoraConfig(
                    r=self.args.lora_text_encoder_r,
                    lora_alpha=self.args.lora_text_encoder_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=self.args.lora_text_encoder_dropout,
                    bias=self.args.lora_text_encoder_bias
                )
                self.text_encoder = get_peft_model(self.text_encoder, config)
                self.text_encoder.print_trainable_parameters()
            # If not using LoRA, text_encoder remains trainable
        else:
            self.text_encoder.requires_grad_(False) # Freeze text encoder

        if self.args.use_lora:
            print("Applying LoRA to UNet...")
            config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=["to_q", "to_v", "query", "value"],
                lora_dropout=self.args.lora_dropout,
                bias=self.args.lora_bias
            )
            self.unet = get_peft_model(self.unet, config)
            self.unet.print_trainable_parameters()

        # --- Apply Optimizations ---
        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed.")

        if self.args.gradient_checkpointing:
            print("Enabling gradient checkpointing...")
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder and not self.args.use_lora:
                self.text_encoder.gradient_checkpointing_enable()

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Configures the AdamW optimizer."""
        
        params_to_optimize: Iterable[torch.nn.Parameter]
        if self.args.train_text_encoder:
            params_to_optimize = itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
        else:
            params_to_optimize = self.unet.parameters()

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
                print("Using 8-bit AdamW optimizer.")
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install bitsandbytes: `pip install bitsandbytes`"
                )
        else:
            optimizer_class = torch.optim.AdamW
            print("Using standard AdamW optimizer.")

        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon
        )
        return optimizer

    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """ConfigGures the learning rate scheduler."""
        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power
        )
        return lr_scheduler

    @staticmethod
    def save_weights(
        accelerator: Accelerator,
        unet: UNet2DConditionModel,
        text_encoder: CLIPTextModel,
        output_dir: Path,
        global_step: int,
        save_text_encoder: bool
    ):
        """Saves the LoRA weights for the UNet and optionally the Text Encoder."""
        
        ckpt_dir = output_dir / f'checkpoint-{global_step}'
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        # 1. Save UNet
        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_dir = ckpt_dir / 'unet'
        unwrapped_unet.save_pretrained(unet_dir, state_dict=accelerator.get_state_dict(unet))

        # 2. Save Text Encoder (if trained)
        if save_text_encoder and text_encoder is not None:
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            textenc_dir = ckpt_dir / 'text_encoder'
            textenc_state = accelerator.get_state_dict(text_encoder)
            unwrapped_text_encoder.save_pretrained(textenc_dir, state_dict=textenc_state)
            
        print(f"Saved LoRA weights to {ckpt_dir}")
