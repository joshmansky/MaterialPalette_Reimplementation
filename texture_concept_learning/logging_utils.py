"""
Re-implementation of the logger setup.
"""

import logging
from argparse import Namespace
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger

def setup_logging(args: Namespace, accelerator: Accelerator) -> logging.Logger:
    """Configures the accelerator logger and optional wandb."""
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=False)

    # Set verbosity for different libraries
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    # Initialize wandb if requested
    if args.report_to == 'wandb' and accelerator.is_main_process:
        try:
            import wandb
            wandb.login(key=args.wandb_key)
            wandb.init(project=args.wandb_project_name, config=vars(args))
            print("Wandb logging enabled and initialized.")
        except ImportError:
            logger.warning("Wandb not installed. Please install: `pip install wandb`")
        except Exception as e:
            logger.error(f"Could not initialize wandb: {e}")

    return logger
