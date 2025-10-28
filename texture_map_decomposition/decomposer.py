"""
Re-implemented MaterialDecomposer class.

This class provides a high-level API to load the pre-trained
decomposition model and run inference on a single texture image.
"""

import torch
from PIL import Image
from pathlib import Path
from easydict import EasyDict

# Import our re-implemented modules
from . import network
from . import utils

class MaterialDecomposer:
    """
    A wrapper for the MaterialPalette SVBRDF decomposition model.
    """
    
    def __init__(self, weights_path: str | Path, device: str = None):
        """
        Initializes and loads the decomposition model.

        Args:
            weights_path: Path to the .pth or .ckpt PyTorch Lightning checkpoint.
            device: The torch device to use (e.g., "cuda", "cpu").
                    Autodetects if None.
        """
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")

        # 1. Build the model "shell" from our network.py
        self.model = network.build_decomposition_net()
        
        # 2. Load the weights from the PL checkpoint
        self._load_weights_from_checkpoint(weights_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 3. Get the required data transforms
        self.transforms = utils.get_inference_transforms()
        print("MaterialDecomposer initialized successfully.")

    def _load_weights_from_checkpoint(self, ckpt_path: str | Path):
        """
        Loads model weights from a PyTorch Lightning checkpoint file.
        
        This method intelligently filters the state_dict to extract
        only the `model.*` weights, making it independent of the
        original `Vanilla` LightningModule class.
        """
        try:
            full_checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            # PL checkpoints store weights under the "state_dict" key
            if "state_dict" not in full_checkpoint:
                raise KeyError("Checkpoint is not a valid PyTorch Lightning file (missing 'state_dict').")
                
            full_state_dict = full_checkpoint["state_dict"]
            
            # Filter for `model.*` keys and remove the "model." prefix
            model_state_dict = {
                k.replace("model.", "", 1): v
                for k, v in full_state_dict.items()
                if k.startswith("model.")
            }
            
            if not model_state_dict:
                raise RuntimeError("No 'model.*' weights found in the checkpoint state_dict.")

            # Load the filtered weights into our model shell
            self.model.load_state_dict(model_state_dict)
            
        except Exception as e:
            print(f"Error loading weights from {ckpt_path}: {e}")
            raise

    @torch.no_grad()
    def decompose(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        output_filename_base: str = None
    ):
        """
        Runs the full decomposition pipeline on a single image.

        Args:
            image_path: Path to the generated texture.
            output_dir: Folder to save the output PBR maps.
            output_filename_base: Base name for files (e.g., "my_texture").
                                  If None, uses the image's filename.
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        
        if not output_filename_base:
            output_filename_base = image_path.stem

        print(f"Decomposing {image_path.name}...")

        # 1. Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return
            
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        # 2. Run inference
        # Model returns an EasyDict of raw outputs
        raw_output = self.model(tensor)
        
        # 3. Post-process to get [0, 1] maps
        processed_maps = utils.post_process_maps(raw_output)
        
        # 4. Save the final maps
        utils.save_pbr_maps(
            processed_maps,
            output_dir,
            output_filename_base
        )
        
        print(f"Successfully saved PBR maps to {output_dir}")
