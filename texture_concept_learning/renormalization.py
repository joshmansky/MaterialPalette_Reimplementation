"""
renormalization.py (Re-implemented)

A module for color-matching a generated image to a source reference.

This implementation uses a class-based approach. A `Renormalizer` object
is first initialized with a source image and its corresponding mask.
It calculates the 'stable' color statistics from this reference.
Its `normalize_image` method can then be called on any target image
to match its color profile to the stored reference.
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as tf
from torch import Tensor

class Renormalizer:
    """
    Manages color statistics of a reference image and applies
    them to new target images.
    """
    
    def __init__(
        self,
        source_image_path: Path,
        mask_path: Path,
        outlier_percentile: float = 0.5,
        device: torch.device = None
    ):
        """
        Initializes the normalizer by calculating the stable statistics
        from the source image and mask.

        Args:
            source_image_path: Path to the original, high-res source photo.
            mask_path: Path to the corresponding segmentation mask.
            outlier_percentile: Percentile of high/low pixels to exclude
                                from statistics (e.g., 0.5 means 0.5% low
                                and 0.5% high are ignored).
            device: The torch device to use for computation.
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.mean_ref = None
        self.std_ref = None
        
        self._calculate_reference_stats(
            source_image_path, mask_path, outlier_percentile
        )

    def _calculate_reference_stats(
        self,
        source_path: Path,
        mask_path: Path,
        percentile: float
    ):
        """Internal method to load images and compute stable stats."""
        
        # 1. Load source image
        pil_img = Image.open(source_path).convert('RGB')
        source_tensor = tf.to_tensor(pil_img).to(self.device)
        
        # 2. Load and prepare mask
        pil_mask = Image.open(mask_path).convert('L')
        # Resize mask to match source image
        mask_tensor = tf.to_tensor(
            pil_mask.resize(pil_img.size, Image.NEAREST)
        ).to(self.device)
        
        mask_bool = (mask_tensor > 0.5).squeeze(0) # [H, W]
        
        if mask_bool.sum() == 0:
            print(f"Warning: Mask {mask_path.name} is empty. Using stats from a full image.")
            # Fallback: use full image stats
            self.mean_ref = source_tensor.mean(dim=(1, 2), keepdim=True)
            self.std_ref = source_tensor.std(dim=(1, 2), keepdim=True)
            return

        # 3. Get stable pixel thresholds using NumPy for a different implementation
        gray_tensor = tf.rgb_to_grayscale(source_tensor)
        masked_gray_pixels = gray_tensor[mask_bool[None, ...]] # Flattened 1D tensor
        
        # Use numpy for percentile calculation
        gray_np = masked_gray_pixels.cpu().numpy()
        low_thresh = np.percentile(gray_np, percentile)
        high_thresh = np.percentile(gray_np, 100.0 - percentile)
        
        # 4. Create the final "stable" mask
        stable_mask = (gray_tensor >= low_thresh) & \
                      (gray_tensor <= high_thresh) & \
                      mask_bool[None, ...] # [1, H, W]
        
        # 5. Calculate mean/std from *color* pixels within this stable region
        # Permute to [H, W, C] for easier boolean mask indexing
        stable_pixels_rgb = source_tensor.permute(1, 2, 0)[stable_mask.squeeze(0)]
        
        if stable_pixels_rgb.numel() == 0:
            print(f"Warning: No stable pixels found for {mask_path.name}. Using all masked pixels.")
            # Fallback: use all pixels under the mask
            stable_pixels_rgb = source_tensor.permute(1, 2, 0)[mask_bool.squeeze(0)]

        # Calculate stats (dim=0 is the pixel dimension)
        # Reshape to [1, C, 1, 1] for broadcasting
        self.mean_ref = stable_pixels_rgb.mean(dim=0).view(1, 3, 1, 1)
        self.std_ref = stable_pixels_rgb.std(dim=0).view(1, 3, 1, 1)

    def normalize_image(
        self,
        target_image_path: Path,
        output_path: Path
    ) -> Path:
        """
        Loads a target image, normalizes it using the stored reference
        statistics, and saves it to the output path.
        """
        if self.mean_ref is None or self.std_ref is None:
            raise RuntimeError("Renormalizer has not been properly initialized.")
            
        # 1. Load target image
        pil_target = Image.open(target_image_path).convert('RGB')
        target_tensor = tf.to_tensor(pil_target)[None, ...].to(self.device)
        
        # 2. Get target image stats
        mean_gen = target_tensor.mean(dim=(2, 3), keepdim=True)
        std_gen = target_tensor.std(dim=(2, 3), keepdim=True)
        
        # 3. Apply normalization
        # Add epsilon to std_gen to prevent division by zero
        normalized_tensor = (target_tensor - mean_gen) / (std_gen + 1e-6) \
                            * self.std_ref + self.mean_ref
                            
        # 4. Clamp and save
        normalized_tensor.clamp_(0, 1)
        
        tf.to_pil_image(normalized_tensor.squeeze(0)).save(output_path)
        return output_path

def main():
    """Provides a command-line interface for the Renormalizer."""
    
    parser = argparse.ArgumentParser(
        description="Re-normalize a generated image to match a source."
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Path to the original source (reference) image."
    )
    parser.add_argument(
        "--mask", type=Path, required=True,
        help="Path to the mask for the source image."
    )
    parser.add_argument(
        "--target", type=Path, required=True,
        help="Path to the generated image you want to normalize."
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to save the new, normalized image."
    )
    parser.add_argument(
        "--percentile", type=float, default=0.5,
        help="Outlier percentile to exclude (default: 0.5)."
    )
    args = parser.parse_args()

    print(f"Initializing Renormalizer with {args.source.name}...")
    try:
        normalizer = Renormalizer(
            source_image_path=args.source,
            mask_path=args.mask,
            outlier_percentile=args.percentile
        )
        
        print(f"Normalizing {args.target.name}...")
        normalizer.normalize_image(
            target_image_path=args.target,
            output_path=args.output
        )
        print(f"Successfully saved normalized image to {args.output}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()