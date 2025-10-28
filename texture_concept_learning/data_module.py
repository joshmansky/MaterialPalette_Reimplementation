"""
Re-implementation of the DreamBooth data pipeline.

This module provides a `DreamBoothDataModule` class that encapsulates
all data-related logic, including:
- Loading the tokenizer.
- Creating the `_DreamBoothDataset`.
- Providing a `train_dataloader` for the training script.
"""

from pathlib import Path
from PIL import Image
from typing import Tuple, Optional
from argparse import Namespace

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPTokenizer

class _DreamBoothDataset(Dataset):
    """
    Internal dataset class to load and transform images and prompts.
    (This logic is from the original `data.py`)
    """

    def __init__(
        self,
        data_dir: Path,
        prompt: str,
        tokenizer: CLIPTokenizer,
        size: int
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.prompt = prompt

        self.instance_images_path = [
            f for f in data_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ]

        if not self.instance_images_path:
            raise ValueError(f"No images found in directory: {data_dir}")

        self.image_transforms = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.instance_images_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one item from the dataset.

        Returns:
            A tuple of (image_tensor, prompt_input_ids).
        """

        # Load image
        image_path = self.instance_images_path[index % len(self)]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Apply transforms
        img_tensor = self.image_transforms(image)

        # Tokenize prompt
        prompt_ids = self.tokenizer(
            self.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0] # Get the first (and only) item

        return img_tensor, prompt_ids


class DreamBoothDataModule:
    """
    Encapsulates all data preparation logic for the LoRA finetuning.
    """

    def __init__(self, args: Namespace):
        """
        Initializes the data module.

        Args:
            args: The global Namespace object containing all settings.
        """
        self.data_dir = Path(args.data_dir)
        self.prompt = args.prompt
        self.size = args.resolution
        self.batch_size = args.train_batch_size
        self.num_workers = args.dataloader_num_workers

        self.tokenizer = self._load_tokenizer(args)
        self.train_dataset: Optional[_DreamBoothDataset] = None

    def _load_tokenizer(self, args: Namespace) -> CLIPTokenizer:
        """Loads the tokenizer from the specified path."""
        tokenizer_path_or_name: str
        subfolder: Optional[str] = None

        if args.tokenizer_name:
            # If a specific tokenizer name/path is given, use it directly
            tokenizer_path_or_name = args.tokenizer_name
        elif args.pretrained_model_name_or_path:
            # Otherwise, assume the tokenizer is in a subfolder of the main model
            tokenizer_path_or_name = args.pretrained_model_name_or_path
            subfolder = "tokenizer"
        else:
            # This case should ideally be caught by arg parsing if required
             raise ValueError("Either tokenizer_name or pretrained_model_name_or_path must be provided.")

        print(f"Loading tokenizer '{tokenizer_path_or_name}'" + (f" from subfolder '{subfolder}'" if subfolder else ""))

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path_or_name,
                subfolder=subfolder,
                revision=args.revision,
                use_fast=False
            )
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

    def setup(self):
        """Creates the training dataset."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Instance images directory not found: {self.data_dir}")

        print(f"Loading dataset from: {self.data_dir}")
        self.train_dataset = _DreamBoothDataset(
            data_dir=self.data_dir,
            prompt=self.prompt,
            tokenizer=self.tokenizer,
            size=self.size
        )
        print(f"Found {len(self.train_dataset)} training images.")

    def train_dataloader(self) -> DataLoader:
        """Returns the configured DataLoader for training."""
        if self.train_dataset is None:
            self.setup()

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

