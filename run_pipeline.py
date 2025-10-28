"""
run_pipeline.py

End-to-end execution script for the re-implemented MaterialPalette pipeline.

This script automates the process for a single input image:
1. Prepares the input data and generates an all-white mask.
2. Generates training crops.
3. Trains a DreamBooth LoRA concept model.
4. Generates a seamless, tileable texture using the LoRA.
5. Decomposes the texture into Albedo, Normal, and Roughness maps.

Assumes:
- This script is located in the root of the repository.
- 'example_input.png' is present in the root directory.
- 'decomposition_model.ckpt' (pre-trained weights) is present in the root directory.
- All necessary Python packages are installed.
"""

import sys
import shutil
from pathlib import Path
from PIL import Image
import warnings

# --- Configuration ---

# Input Image
INPUT_IMAGE_FILENAME = "example_input.png" # Must be in the root directory

# Mask & Concept Details
MASK_NAME = "example_texture" # Base name for mask file and directories
CONCEPT_TOKEN = "example_sks_texture" # Unique token for DreamBooth

# Directory Names
DATA_PREP_DIR = Path("material_data_input") # Temp dir for data prep
WEIGHTS_OUTPUT_DIR = Path("concept_weights") # Where LoRA weights are saved
GENERATED_TEXTURE_DIR = Path("generated_textures") # Where the final texture is saved
PBR_MAPS_OUTPUT_DIR = Path("pbr_material_maps") # Where final PBR maps are saved

# Model Weights
DECOMPOSITION_WEIGHTS_FILENAME = "model.ckpt" # Must be in the root

# LoRA Training Settings (Override defaults from lora_finetuning.py if needed)
LORA_TRAINING_OVERRIDES = {
    "max_train_steps": 800, # using 800 for good quality
    "checkpointing_steps": 800,
    "learning_rate": 1e-4,
    "seed": 42,
    # Add other parameters from DEFAULT_CONFIG in lora_finetuning.py here if needed
}

# Texture Generation Settings
TEXTURE_GENERATION_OVERRIDES = {
    "resolution": 1024,
    "prompt_key": 'p1', # 'top view realistic texture of SKS'
    "seed": 42,
    "num_inference_steps": 50,
    # Add other parameters from DEFAULT_GENERATION_CONFIG in tileable_texture_creation.py
}

# --- End Configuration ---


def prepare_input_data(
    root_dir: Path,
    input_image_path: Path,
    mask_base_name: str
) -> Path:
    """Creates the directory structure and the all-white mask."""
    print("--- Preparing Input Data ---")
    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    material_dir = root_dir / mask_base_name
    mask_dir = material_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Load the source image
    try:
        img = Image.open(input_image_path).convert("RGB")
    except Exception as e:
        raise IOError(f"Could not open or read input image: {input_image_path}. Error: {e}")

    # 1. Copy the source image to the material folder
    shutil.copyfile(input_image_path, material_dir / input_image_path.name)
    print(f"Copied source image to: {material_dir / input_image_path.name}")

    # 2. Create and save the all-white mask
    mask = Image.new("L", img.size, 255) # 255 = all white
    mask_path = mask_dir / f"{mask_base_name}.png"
    mask.save(mask_path)
    print(f"Created all-white mask at: {mask_path}")

    return material_dir


def main():
    """Runs the full MaterialPalette pipeline."""
    warnings.filterwarnings("ignore") # Suppress common library warnings
    root_dir = Path(".") # Assume script is run from the repo root

    # --- Check for required files ---
    input_image_path = root_dir / INPUT_IMAGE_FILENAME
    decomposition_weights_path = root_dir / DECOMPOSITION_WEIGHTS_FILENAME
    if not input_image_path.exists():
        print(f"ERROR: Input image '{INPUT_IMAGE_FILENAME}' not found in the root directory.")
        sys.exit(1)
    if not decomposition_weights_path.exists():
        print(f"ERROR: Decomposition weights '{DECOMPOSITION_WEIGHTS_FILENAME}' not found in the root directory.")
        print("Please download it from the original MaterialPalette repository releases.")
        sys.exit(1)

    # --- Dynamically import our modules ---
    # This allows running the script from the root
    try:
        from texture_concept_learning.generate_crops import run_cropping
        from texture_concept_learning.lora_finetuning import invert
        from texture_concept_learning.tileable_texture_creation import generate_texture_from_lora
        from texture_map_decomposition.decomposer import MaterialDecomposer
    except ImportError as e:
        print(f"ERROR: Could not import necessary modules. {e}")
        print("Please ensure the script is run from the root of the repository")
        print("and the 'texture_concept_learning' and 'texture_map_decomposition' folders exist.")
        sys.exit(1)

    print("--- Starting MaterialPalette Re-implementation Pipeline ---")

    # === Step 1: Prepare Input Data ===
    try:
        material_input_dir = prepare_input_data(
            root_dir=DATA_PREP_DIR,
            input_image_path=input_image_path,
            mask_base_name=MASK_NAME
        )
    except Exception as e:
        print(f"\nERROR during data preparation: {e}")
        sys.exit(1)

    # === Step 2: Generate Training Crops ===
    print("\n--- Generating Training Crops ---")
    try:
        run_cropping(
            data_path=material_input_dir,
            patch_sizes=[512, 256],
            threshold=0.9,
            topk=50
        )
        training_data_dir = material_input_dir / "crops" / MASK_NAME
        if not training_data_dir.exists() or not any(training_data_dir.iterdir()):
             raise RuntimeError("Crop generation failed or produced no images.")
        print(f"Training crops generated at: {training_data_dir}")
    except Exception as e:
        print(f"\nERROR during crop generation: {e}")
        sys.exit(1)

    # === Step 3: Run LoRA Finetuning ===
    print("\n--- Training LoRA Concept Model ---")
    try:
        lora_checkpoint_path = invert(
            data_dir=training_data_dir,
            prompt_token=CONCEPT_TOKEN,
            output_root_dir=WEIGHTS_OUTPUT_DIR,
            **LORA_TRAINING_OVERRIDES # Pass overrides
        )
        print(f"LoRA training complete. Weights saved in: {lora_checkpoint_path}")
    except Exception as e:
        print(f"\nERROR during LoRA training: {e}")
        sys.exit(1)


    # === Step 4: Generate Seamless Texture ===
    print("\n--- Generating Seamless Texture ---")
    try:
        generated_texture_path = generate_texture_from_lora(
            lora_path=lora_checkpoint_path,
            output_dir=GENERATED_TEXTURE_DIR,
            prompt_token=CONCEPT_TOKEN,
            **TEXTURE_GENERATION_OVERRIDES # Pass overrides
        )
        if not generated_texture_path.exists():
            raise RuntimeError("Texture generation failed.")
        print(f"Seamless texture generated: {generated_texture_path}")
    except Exception as e:
        print(f"\nERROR during texture generation: {e}")
        sys.exit(1)

    # === Step 5: Run PBR Decomposition ===
    print("\n--- Decomposing Texture into PBR Maps ---")
    try:
        decomposer = MaterialDecomposer(weights_path=decomposition_weights_path)
        decomposer.decompose(
            image_path=generated_texture_path,
            output_dir=PBR_MAPS_OUTPUT_DIR,
            output_filename_base=f"{MASK_NAME}_final" # e.g., example_texture_final_albedo.png
        )
        print(f"PBR maps saved in: {PBR_MAPS_OUTPUT_DIR}")
    except Exception as e:
        print(f"\nERROR during PBR decomposition: {e}")
        sys.exit(1)

    print("\n--- MaterialPalette Pipeline Finished Successfully! ---")
    print(f"Input Image: {input_image_path}")
    print(f"Generated Texture: {generated_texture_path}")
    print(f"Output PBR Maps (Albedo, Normal, Roughness) in: {PBR_MAPS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
