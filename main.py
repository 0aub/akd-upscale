import argparse
import json
import os

import torch
import random
import numpy as np

from src.logger import Logger
from src.teacher import TeacherUpscaler
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
    parser = argparse.ArgumentParser(description="Knowledge Distillation with DIV2K + Stable Diffusion Upscaler")
    
    # Basic Dataset Paths
    parser.add_argument("--low_resolution_folder", type=str, default="data/all_lr_images",
                        help="Folder containing ALL low-res images.")
    parser.add_argument("--teacher_folder", type=str, default="data/teacher_upscaled",
                        help="Folder where teacher outputs will be saved (split into train/val/test subfolders).")
    parser.add_argument("--overwrite_teacher_data", action="store_true", default=False)

    # Teacher Model Info
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-x4-upscaler",
                        help="HuggingFace model ID for the upscaler pipeline.")
    parser.add_argument("--teacher_prompt", type=str, default="a photo")
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    
    # Testing
    parser.add_argument("--test_path", type=str, default="data/test/low_res")
    parser.add_argument("--test_count", type=int, default=20)

    # Teacher Model Info
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-x4-upscaler")
    parser.add_argument("--teacher_prompt", type=str, default="a photo")
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    
    # Generator parameters
    parser.add_argument("--generator_up_factor", type=int, default=4, help="Upscaling factor for the generator")
    parser.add_argument("--generator_base_channels", type=int, default=32, help="Base number of channels in the generator")
    parser.add_argument("--generator_num_blocks", type=int, default=4, help="Number of residual blocks in the generator")

    # Discriminator parameters
    parser.add_argument("--discriminator_in_channels", type=int, default=3, help="Number of input channels for the discriminator")
    parser.add_argument("--discriminator_base_channels", type=int, default=64, help="Base number of channels in the discriminator")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
        
    # Experiment / Logging
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--log_path", type=str, default="log")

    # Loss Function
    parser.add_argument("--generator_loss", type=parse_loss_weights, default={"vgg":1.0, "l1":0.1},
                        help="Loss weights as a JSON string, e.g., '{\"vgg\": 1.0, \"l1\": 0.1}'")
    parser.add_argument("--discriminator_loss", type=parse_loss_weights, default={"vgg":1.0, "l1":0.1},
                        help="Loss weights as a JSON string, e.g., '{\"vgg\": 1.0, \"l1\": 0.1}'")
    
    # Generator Optimizer
    parser.add_argument("--generator_optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop", "AdamW"])
    parser.add_argument("--generator_learning_rate", type=float, default=1e-4)
    parser.add_argument("--generator_weight_decay", type=float, default=0.0)
    parser.add_argument("--generator_betas", type=float, nargs=2, default=None)  # Two floats for betas
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

    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    cfg = parse_arguments()
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Create logger
    logger = Logger(
        log_path=cfg.log_path,
        exp_name=cfg.exp_name,
        save=True  # saving logs to file
    )
    # Log configuration
    logger.logline()
    logger.log_config(vars(cfg))

    logger.logline()
    trainer = Trainer(cfg, logger)
    
    # 1) Prepare LR images from HR
    logger.logline()
    logger.log("[Teacher]  Generating Teacher Outputs")
    upscaler = TeacherUpscaler(cfg, logger)
    upscaler.run() 
        
    # 3) Train the student model
    logger.logline()
    logger.log("[Student]  Student Adversarial Knowledge Distillation Training")
    trainer.train()

    # 4) Train the student model
    logger.logline()
    logger.log("[Student]  Student and Teacher Testing")
    trainer.test()
    
    # 5) Save the final student model
    trainer.save_model(os.path.join(logger.exp_path, "model.pth"))
    

if __name__ == "__main__":
    main()
