import argparse
import json
import os

import torch
import random
import numpy as np

from src.logger import Logger
from src.teacher import Teacher
from src.trainer import Trainer


def parse_loss_weights(value):
    """
    Parse a string like '{"vgg": 1.0, "l1": 0.1}' into a dictionary.
    """
    try:
        # Convert the string input to a dictionary using json.loads
        loss_weights = json.loads(value)
        # Validate that the values are floats or can be converted to floats
        if not isinstance(loss_weights, dict) or not all(isinstance(v, (int, float)) for v in loss_weights.values()):
            raise ValueError
        return loss_weights
    except (json.JSONDecodeError, ValueError):
        raise argparse.ArgumentTypeError("Loss weights must be a valid JSON string, e.g., '{\"vgg\": 1.0, \"l1\": 0.1}'")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Knowledge Distillation + Stable Diffusion Upscaler")
    
    # Basic Dataset Paths
    parser.add_argument("--low_resolution_folder", type=str, default="data/all_lr_images", help="Folder containing ALL low-res images.")
    parser.add_argument("--teacher_folder", type=str, default="data/teacher_upscaled", help="Folder where teacher outputs will be saved (split into train/val/test subfolders).")
    parser.add_argument("--overwrite_teacher_data", action="store_true", default=False)
    
    # Teacher Model Info
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-x4-upscaler")
    parser.add_argument("--teacher_prompt", type=str, default="a photo")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    
    # Generator parameters
    parser.add_argument("--generator_up_factor", type=int, default=4, help="Upscaling factor for the generator")
    parser.add_argument("--generator_base_channels", type=int, default=32, help="Base number of channels in the generator")
    parser.add_argument("--generator_attention", type=str, default=None, help="Attention type for the generator (e.g. 'se_layer', 'cbam', etc.)")
    parser.add_argument("--generator_block_type", type=str, default="unet", help="Blocks type for the generator ('unet' or 'rdb')")
    parser.add_argument("--generator_advanced_upsampling", action="store_true", default=False, help="Use advanced upsampling refinement block in the generator")
    parser.add_argument("--generator_rdb_num_layers", type=int, default=4, help="Number of layers in each RDB block (if using RDB blocks)")
    parser.add_argument("--generator_rdb_growth_rate", type=int, default=16, help="Growth rate for each RDB block (if using RDB blocks)")
    
    # Discriminator parameters
    parser.add_argument("--discriminator_in_channels", type=int, default=3, help="Number of input channels for the discriminator")
    parser.add_argument("--discriminator_base_channels", type=int, default=64, help="Base number of channels in the discriminator")
    parser.add_argument("--discriminator_attention", type=str, default=None, help="Attention type for the discriminator (e.g. 'se_layer', 'cbam', etc.)")
    
    # Discriminator training stability parameters
    parser.add_argument("--discriminator_updates", type=int, default=1, help="Number of discriminator updates per generator update")
    parser.add_argument("--use_gradient_penalty", action="store_true", help="Enable gradient penalty for the discriminator")
    parser.add_argument("--gradient_penalty_weight", type=float, default=10.0, help="Weight for the gradient penalty term")
    parser.add_argument("--label_smoothing", type=float, default=0.9, help="Label smoothing value for real images in discriminator training")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
        
    # Experiment / Logging
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--log_path", type=str, default="log")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--test_freq", type=int, default=None)

    # Resume Training
    parser.add_argument("--resume", action="store_true", help="Resume training with full state (models + optimizers + training metadata)")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune models from checkpoint (load only models, ignore optimizers/training state)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (required for finetuning, optional for resume)")

    # Loss Function
    parser.add_argument("--generator_loss", type=parse_loss_weights, default={"vgg":1.0, "l1":0.1}, help="Loss weights as a JSON string, e.g., '{\"vgg\": 1.0, \"l1\": 0.1}'")
    parser.add_argument("--discriminator_loss", type=parse_loss_weights, default={"vgg":1.0, "l1":0.1}, help="Loss weights as a JSON string, e.g., '{\"vgg\": 1.0, \"l1\": 0.1}'")
    parser.add_argument("--adversarial_weight", type=float, default=1.0)
    
    # Generator Optimizer
    parser.add_argument("--generator_optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop", "AdamW"])
    parser.add_argument("--generator_learning_rate", type=float, default=1e-4)
    parser.add_argument("--generator_weight_decay", type=float, default=0.0)
    parser.add_argument("--generator_betas", type=float, nargs=2, default=None)
    parser.add_argument("--generator_clip_max_norm", type=float, default=1.0)

    # Student Generator Scheduler
    parser.add_argument("--generator_scheduler", type=str, default=None,
                        choices=["step", "exponential", "cosine", "linear", "constant", "constant_with_warmup", None])
    parser.add_argument("--generator_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--generator_warmup_steps", type=int, default=0)

    # Student Discriminator Optimizer
    parser.add_argument("--discriminator_optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop", "AdamW"])
    parser.add_argument("--discriminator_learning_rate", type=float, default=1e-4)
    parser.add_argument("--discriminator_weight_decay", type=float, default=0.0)
    parser.add_argument("--discriminator_betas", type=float, nargs=2, default=None)  # Two floats for betas
    parser.add_argument("--discriminator_clip_max_norm", type=float, default=1.0)

    # Discriminator Scheduler
    parser.add_argument("--discriminator_scheduler", type=str, default=None,
                        choices=["step", "exponential", "cosine", "linear", "constant", "constant_with_warmup", None])
    parser.add_argument("--discriminator_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--discriminator_warmup_steps", type=int, default=0)
    
    # Seed and Device
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--device", type=str, default="cuda")


    # inference-only arguments
    parser.add_argument("--inference_only", action="store_true", default=False, help="Skip teacher data generation and training. Load a checkpoint and run inference only.")
    parser.add_argument("--inference_input_path", type=str, default=None, help="Path to a single image, a directory of images, or a single video for inference.")
    parser.add_argument("--inference_output_path", type=str, default=None, help="Path to output file (for single image/video) or folder (for a directory of images).")


    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    cfg = parse_arguments()

    if cfg.inference_only:
        if cfg.inference_input_path is None or cfg.inference_output_path is None:
            raise ValueError("For inference_only, must specify both --inference_input_path and --inference_output_path.")

        logger = Logger(
            log_path=cfg.log_path,
            exp_name=cfg.exp_name,
            save=False,
            checkpoint=cfg.checkpoint,
            resume=False,
            finetune=False
        )

        trainer = Trainer(cfg, logger)

        trainer.inference(cfg.inference_input_path, cfg.inference_output_path)
        return 
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Create logger
    logger = Logger(
        log_path=cfg.log_path,
        exp_name=cfg.exp_name,
        save=True,
        checkpoint=cfg.checkpoint,
        resume=cfg.resume,
        finetune=cfg.finetune
    )
    # Log configuration
    logger.logline()
    logger.log_config(vars(cfg))
    
    # 1) Prepare LR images from HR
    logger.logline()
    logger.log("[Teacher]  Preparing Teacher Outputs")
    upscaler = Teacher(cfg, logger)
    upscaler.run() 
        
    # 3) Train the student model
    logger.logline()
    logger.log("[Student]  Student Adversarial Knowledge Distillation Training")
    trainer = Trainer(cfg, logger)
    trainer.train()

    # 4) Train the student model
    logger.logline()
    logger.log("[Student]  Student and Teacher Testing")
    trainer.test()
        

if __name__ == "__main__":
    main()
