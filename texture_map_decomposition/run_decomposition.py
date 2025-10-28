"""
Command-Line Interface for SVBRDF Decomposition.

This script provides a simple way to run the MaterialDecomposer
on a texture image from your terminal.
"""

import argparse
from pathlib import Path
import sys

# Ensure the local 'capture' directory is on the path
# if running this script directly.
sys.path.append(str(Path(__file__).parent.parent))

try:
    from capture.decomposer import MaterialDecomposer
except ImportError:
    print("Error: Could not import MaterialDecomposer.")
    print("Please ensure you are running this from the project's root directory,")
    print("or that the 'capture' module is in your PYTHONPATH.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Decompose a texture image into its PBR maps "
                    "(Albedo, Normals, Roughness)."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the generated seamless texture PNG file."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the pre-trained '.ckpt' or '.pth' model weights file."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the output albedo.png, normal.png, etc."
    )
    parser.add_warning = parser.add_argument
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Base name for output files. (Default: uses the input image name)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda', 'cpu'). (Default: autodetect)"
    )
    args = parser.parse_args()

    try:
        # 1. Initialize the decomposer
        decomposer = MaterialDecomposer(
            weights_path=args.weights,
            device=args.device
        )
        
        # 2. Run the decomposition
        decomposer.decompose(
            image_path=args.image,
            output_dir=args.output_dir,
            output_filename_base=args.name
        )
        
    except Exception as e:
        print(f"\nAn error occurred during decomposition: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
