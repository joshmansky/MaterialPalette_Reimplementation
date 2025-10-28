"""
Utilities for the decomposition model.

Includes:
- Data transforms for inference.
- Post-processing functions to convert raw model output to maps.
- Helper functions to save the final PBR map images.

Logic is derived from the original `source/routine.py`.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from easydict import EasyDict
from pathlib import Path

# These are the standard ImageNet stats, confirmed from
# `source/routine.py:Vanilla.__init__`
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

def get_inference_transforms() -> T.Compose:
    """
    Returns the exact transforms the model was trained with.
    From `source/routine.py`: it's just ToTensor and Normalize.
    The original model seems to handle arbitrary resolutions.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ])

# --- Post-Processing Functions (from source/routine.py) ---

def encode_as_unit_interval(x: torch.Tensor) -> torch.Tensor:
    """
    Maps a tensor from [-1, 1] to [0, 1].
    This is the inverse of the `tanh` activation.
    """
    return (x * 0.5) + 0.5

def gamma_encode(x: torch.Tensor) -> torch.Tensor:
    """
    Applies standard gamma correction.
    The original code imports this; we assume standard sRGB gamma 2.2.
    """
    return torch.pow(x.clamp(0, 1), 1.0 / 2.2)

def post_process_maps(raw_output: EasyDict) -> EasyDict:
    """
    Applies the post-processing logic from `source/routine.py:post_process_`
    to convert raw model outputs into final, interpretable maps.

    Args:
        raw_output: The EasyDict directly from the model
                    (e.g., {'albedo': ..., 'normals': ..., 'roughness': ...})
    
    Returns:
        A new EasyDict with processed [0, 1] range maps.
    """
    processed = EasyDict()
    tanh = torch.nn.Tanh()

    # 1. Albedo: Tanh -> [-1, 1] -> [0, 1]
    a = tanh(raw_output.albedo)
    processed.albedo = encode_as_unit_interval(a)

    # 2. Roughness: Tanh -> [-1, 1] -> [0, 1]
    # The original repeats the 1-channel output to 3 channels for saving.
    r = tanh(raw_output.roughness)
    processed.roughness = encode_as_unit_interval(r.repeat(1, 3, 1, 1))

    # 3. Normals: Model predicts 2 channels (xy)
    # Tanh -> [-1, 1] -> calculate z -> normalize
    nxy = tanh(raw_output.normals)
    
    # Split the 2 channels. Original code multiplies by 3, which seems
    # aggressive and might be a typo for 1.0. Let's stick to the code.
    # nx, ny = torch.split(nxy * 3, split_size_or_sections=1, dim=1)
    
    # Re-reading `source/routine.py`: `split(nxy*3, ...)`
    # This is very strange. A 3-value multiplier on a unit vector component?
    # Let's assume it's `nxy` directly. If results are bad, this is the first
    # place to check.
    # A more standard interpretation is that nxy is already scaled.
    # Let's follow the code literally.
    
    # UPDATE: No, wait. `nxy*3` is then split. `split_size_or_sections=1`.
    # This implies nxy*3 is NOT the final value, but something to be split.
    # This is confusing.
    # Let's look at `source/model.py:MultiHeadDecoder`. `normals=2`.
    # So `o.normals` is [B, 2, H, W].
    # `nxy = tanh(o.normals)` is [B, 2, H, W].
    # `nxy*3` is [B, 2, H, W].
    # `torch.split(nxy*3, split_size_or_sections=1, dim=1)` splits this
    # into two tensors: [B, 1, H, W] and [B, 1, H, W].
    # This confirms the `*3` is likely a scaling factor.
    
    scaled_nxy = nxy * 3.0 # This seems intended to over-sample slopes
    nx, ny = torch.split(scaled_nxy, split_size_or_sections=1, dim=1)
    
    # Create z component (ones) and concatenate
    nz = torch.ones_like(nx)
    n = torch.cat([nx, ny, nz], dim=1)
    
    # Normalize to get a valid unit-vector normal map
    processed.normals = F.normalize(n, dim=1)
    
    return processed

def save_pbr_maps(
    processed_maps: EasyDict,
    output_dir: Path,
    base_filename: str
):
    """
    Saves the processed Albedo, Normals, and Roughness maps to disk.
    Follows the saving logic from `source/routine.py:predict_step`.
    
    Args:
        processed_maps: The EasyDict from `post_process_maps`.
        output_dir: The folder to save into.
        base_filename: The base name for the files (e.g., "my_texture").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Albedo (with Gamma Correction)
    albedo = gamma_encode(processed_maps.albedo.squeeze(0))
    save_image(albedo, output_dir / f"{base_filename}_albedo.png")

    # 2. Save Normals (Mapped from [-1, 1] to [0, 1])
    # The `processed.normals` is still [-1, 1], so we map it.
    normals_to_save = encode_as_unit_interval(processed_maps.normals.squeeze(0))
    save_image(normals_to_save, output_dir / f"{base_filename}_normals.png")
    
    # 3. Save Roughness (Already [0, 1])
    roughness = processed_maps.roughness.squeeze(0)
    save_image(roughness, output_dir / f"{base_filename}_roughness.png")
