#!/usr/bin/env python3
"""
This code is meant to generate random crops from input images & their corresponding segmentation masked for each material present in the image
to be used as the training data, in training of the DreamBooth LoRA on the Stable Diffusion model that learns the textur concept of a given material.

This script expects a directory structure like this:
/path/to/data/
    source_image.jpg
    /masks/
        mask_01.png
        mask_02.png
        ...

It will produce:
/path/to/data/
    /crops/
        /mask_01/
            00000_x512.png
            00001_x512.png
            ...
        /mask_02/
            00000_x256.png
            ...
"""

import argparse
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from pathlib import Path
from random import shuffle
from typing import List, Tuple, Optional

# Define common image extensions to look for
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def patch_image(
    mask_tensor: torch.Tensor,
    patch_size: int,
    stride: int,
    offset_x: int,
    offset_y: int
) -> Tuple[List[float], List[Tuple[int, int, int, int]]]:
    """
    Slides a window over a mask tensor and calculates patch densities.

    Args:
        mask_tensor: The binarized mask tensor (cropped to the object).
        patch_size: The side length of the square patch.
        stride: The step size for the sliding window.
        offset_x: The original x-coordinate of the mask tensor's top-left corner.
        offset_y: The original y-coordinate of the mask tensor's top-left corner.

    Returns:
        A tuple containing (list_of_densities, list_of_bboxes).
        BBoxes are in (x0, y0, x1, y1) format relative to the original image.
    """
    densities = []
    bboxes = []
    # mask_tensor shape is [C, H, W], we want H and W
    _, height, width = mask_tensor.shape

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = mask_tensor[0, y:y + patch_size, x:x + patch_size]
            density = patch.float().mean().item()
            densities.append(density)

            # Add offset back to get coords relative to original image
            bbox = (
                offset_x + x,
                offset_y + y,
                offset_x + x + patch_size,
                offset_y + y + patch_size
            )
            bboxes.append(bbox)
    return densities, bboxes


def process_mask(
    mask_path: Path,
    pil_ref_image: Image.Image,
    output_dir: Path,
    patch_sizes: List[int],
    threshold: float,
    topk: int
) -> bool:
    """
    Processes a single mask file and saves crops from the reference image.

    Returns:
        True if crops were successfully saved, False otherwise.
    """
    
    cluster_name = mask_path.stem
    cluster_dir = output_dir / cluster_name

    if cluster_dir.exists():
        print(f'  > Skipping "{cluster_name}": output directory already exists.')
        return False  # Indicate skipped

    img_width, img_height = pil_ref_image.size

    # --- Refined Mask Processing ---
    try:
        # 1. Open and resize mask to match the source image
        pil_mask = Image.open(mask_path).resize((img_width, img_height))
        
        # 2. Binarize to find the object's bounding box
        # Convert to 'L' (grayscale), then '1' (1-bit binary)
        binarized_mask = pil_mask.convert('L').point(lambda x: 255 if x > 0 else 0, '1')
        main_bbox = binarized_mask.getbbox()

        if not main_bbox:
            print(f'  > Skipping "{cluster_name}": mask is empty.')
            return False
            
        x0, y0, x1, y1 = main_bbox
        
        # 3. Crop the binarized mask to the bbox and convert to tensor
        cropped_bin_mask = binarized_mask.crop(main_bbox)
        mask_tensor = tf.to_tensor(cropped_bin_mask)
        # tf.to_tensor on a '1' mode image creates a [0, 1] tensor directly
        
    except Exception as e:
        print(f'  > ERROR processing mask "{mask_path.name}": {e}')
        return False

    print(f'  > Processing "{cluster_name}"...')

    kept_bboxes = []
    kept_scales = []
    
    for patch_size in patch_sizes:
        # Ensure patch size isn't larger than the cropped mask itself
        if patch_size > (x1 - x0) or patch_size > (y1 - y0):
            continue  # Skip patch size, it's too big for this object

        stride = patch_size // 5  # 80% overlap
        densities, bboxes = patch_image(
            mask_tensor,
            patch_size,
            stride,
            offset_x=x0,
            offset_y=y0
        )

        # Filter patches that meet the density threshold
        kept_local_res = [b for d, b in zip(densities, bboxes) if d >= threshold]

        if not kept_local_res:
            continue  # No patches found at this size, try a smaller one

        shuffle(kept_local_res)
        
        # Only take up to topk patches
        nb_needed = topk - len(kept_bboxes)
        kept_local_res = kept_local_res[:nb_needed]

        kept_bboxes.extend(kept_local_res)
        kept_scales.extend([patch_size] * len(kept_local_res))

        print(f'    {patch_size}x{patch_size}: found {len(kept_local_res)} patches.')

        # Original logic: "only take largest scale"
        # If we found any patches, we break and don't look for smaller sizes.
        if len(kept_local_res) > 0:
            break 
    
    # --- Saving ---
    if len(kept_bboxes) < 2:
        print(f'   ...skipping save, only found {len(kept_bboxes)} valid patches.')
        return False

    cluster_dir.mkdir(parents=True, exist_ok=True)
    for i, (scale, bbox) in enumerate(zip(kept_scales, kept_bboxes)):
        crop_name = cluster_dir / f'{i:0>5}_x{scale}.png'
        # Crop from the *original* source image
        pil_ref_image.crop(bbox).save(crop_name)
        
    print(f'   ...saved {len(kept_bboxes)} patches to "{cluster_dir.name}".')
    return True  # Indicate success


def find_source_image(data_path: Path) -> Optional[Path]:
    """Finds a single image file in the root directory."""
    files = [f for f in data_path.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]
    
    if len(files) == 0:
        print(f'Error: No source image file found in {data_path}')
        return None
    if len(files) > 1:
        print(f'Error: Found multiple image files in {data_path}. Please ensure only one is present.')
        print(f'Files found: {[f.name for f in files]}')
        return None
        
    return files[0]


def run_cropping(
    data_path: Path,
    patch_sizes: List[int],
    threshold: float,
    topk: int
):
    """Main function to orchestrate the cropping process."""
    
    print(f'---- Starting cropping process in: {data_path}')
    
    # 1. Validate paths
    masks_dir = data_path / 'masks'
    if not masks_dir.is_dir():
        print(f'Error: A /masks subdirectory must be present in {data_path}')
        return
        
    output_dir = data_path / 'crops'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Find and load source image
    img_path = find_source_image(data_path)
    if not img_path:
        return
        
    print(f'---- Processing source image: "{img_path.name}"')
    pil_ref = Image.open(img_path).convert('RGB')

    # 3. Find masks
    masks = sorted([f for f in masks_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS])
    if not masks:
        print(f'Error: No masks found in {masks_dir}')
        return
        
    print(f'  Found {len(masks)} masks...')
    
    # 4. Process each mask
    k = 0  # Counter for successfully processed masks
    for mask_file in masks:
        processed = process_mask(
            mask_path=mask_file,
            pil_ref_image=pil_ref,
            output_dir=output_dir,
            patch_sizes=patch_sizes,
            threshold=threshold,
            topk=topk
        )
        if processed:
            k += 1

    print(f'\n---- Finished. Kept crops for {k}/{len(masks)} masks.')
    print(f'     Output saved to: {output_dir}')



# For testing this script directly
def main():
    """Argument parsing and script entry point."""
    
    parser = argparse.ArgumentParser(
        description="Crop high-density patches from a source image using masks.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__  # Use the module docstring as the help footer
    )
    
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the data directory. Should contain one image and a /masks subdir."
    )
    
    parser.add_argument(
        "--patch_sizes",
        type=int,
        nargs='+',
        default=[512, 256, 192, 128, 64],
        help="List of patch sizes to try, from largest to smallest. (default: 512 256 192 128 64)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Mask density threshold (0.0 to 1.0) to keep a patch. (default: 0.99)"
    )
    
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Maximum number of patches to save per mask. (default: 100)"
    )
    
    args = parser.parse_args()
    
    if not args.path.is_dir():
        print(f"Error: The provided path '{args.path}' is not a directory.")
        return

    run_cropping(
        data_path=args.path,
        patch_sizes=args.patch_sizes,
        threshold=args.threshold,
        topk=args.topk
    )

if __name__ == "__main__":
    main()

