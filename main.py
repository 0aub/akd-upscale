import argparse
import os

import torch
import random
import numpy as np

from src.logger import Logger
from src.teacher import prepare_low_res_images, generate_teacher_outputs
from src.trainer import Trainer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Knowledge Distillation with DIV2K + Stable Diffusion Upscaler")
    
    # Basic dataset paths
    parser.add_argument("--train_hr_folder", type=str, default="data/DIV2K_train_HR")
    parser.add_argument("--valid_hr_folder", type=str, default="data/DIV2K_valid_HR")
    parser.add_argument("--train_lr_folder", type=str, default="data/DIV2K_train_LR")
    parser.add_argument("--valid_lr_folder", type=str, default="data/DIV2K_valid_LR")
    parser.add_argument("--train_teacher_folder", type=str, default="data/DIV2K_train_teacher")
    parser.add_argument("--valid_teacher_folder", type=str, default="data/DIV2K_valid_teacher")
    parser.add_argument("--overwrite_teacher_data", action='store_true', default="overwrite generated data by the teacher")
    
    # Teacher Model Info
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-x4-upscaler")
    parser.add_argument("--teacher_prompt", type=str, default="a photo")
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--up_factor", type=int, default=4)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    # Experiment / Logging
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--log_path", type=str, default="log")
    
    # Optimizer & Scheduler
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop", "AdamW"])
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["step", "exponential", "cosine", "linear", "constant", "constant_with_warmup", None])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    
    # Gradient Clipping
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    
    # Seed
    parser.add_argument("--seed", type=int, default=42)
    
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
    logger.log_config(vars(cfg))
    
    # 1) Prepare LR images from HR
    logger.log("[Stage 1]  Preparing Low-Resolution Images")
    prepare_low_res_images(cfg, logger)
    
    # 2) Generate teacher outputs
    logger.log("[Stage 2]  Generating Teacher Outputs")
    generate_teacher_outputs(cfg, logger)
    
    # 3) Train the student model
    logger.log("[Stage 3]  Student Training (Knowledge Distillation)")
    trainer = Trainer(cfg, logger)
    trainer.train()
    
    # 4) Save the final student model
    trainer.save_model(os.path.join(logger.exp_path, "model.pth"))
    logger.log("[Done]  Training complete. Model saved.")

if __name__ == "__main__":
    main()
