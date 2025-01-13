import os
import glob

from PIL import Image
import torch
from torchvision import transforms

from diffusers import StableDiffusionUpscalePipeline


class DownscaleByFactor:
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = w // self.factor
        new_h = h // self.factor
        return img.resize((new_w, new_h), Image.BICUBIC)


def prepare_low_res_images(cfg, logger):
    """
    Downscale HR images by cfg.up_factor to produce LR images.
    Skips files that already exist unless cfg.overwrite_data=True.
    """
    train_hr_folder = cfg.train_hr_folder
    valid_hr_folder = cfg.valid_hr_folder
    
    train_lr_folder = cfg.train_lr_folder
    valid_lr_folder = cfg.valid_lr_folder
    
    os.makedirs(train_lr_folder, exist_ok=True)
    os.makedirs(valid_lr_folder, exist_ok=True)
    
    # Define a custom transform pipeline (example from your snippet)
    downscale_transform = transforms.Compose([
        DownscaleByFactor(cfg.up_factor)
    ])
    
    # Training HR -> LR
    train_hr_paths = glob.glob(os.path.join(train_hr_folder, "*"))
    logger.log(f"[Teacher]  Generating LR from training HR: {train_hr_folder}, found {len(train_hr_paths)} images.")
    
    for i, path in enumerate(train_hr_paths):
        filename = os.path.basename(path)
        lr_output_path = os.path.join(train_lr_folder, filename)
        
        # Skip if file already exists and no overwrite
        if os.path.exists(lr_output_path) and not cfg.overwrite_data:
            continue
        
        img = Image.open(path).convert("RGB")
        lr_img = downscale_transform(img)
        lr_img.save(lr_output_path)
        
        if (i+1) % 50 == 0:
            logger.log(f"[Teacher]  Generating LR {i+1}/{len(train_hr_paths)}")
    
    # Validation HR -> LR
    valid_hr_paths = glob.glob(os.path.join(valid_hr_folder, "*"))
    logger.log(f"[Teacher]  Generating LR from validation HR: {valid_hr_folder}, found {len(valid_hr_paths)} images.")
    
    for i, path in enumerate(valid_hr_paths):
        filename = os.path.basename(path)
        lr_output_path = os.path.join(valid_lr_folder, filename)
        
        # Skip if file already exists and no overwrite
        if os.path.exists(lr_output_path) and not cfg.overwrite_data:
            continue
        
        img = Image.open(path).convert("RGB")
        lr_img = downscale_transform(img)
        lr_img.save(lr_output_path)
        
        if (i+1) % 50 == 0:
            logger.log(f"[Teacher]  Generating LR {i+1}/{len(valid_hr_paths)}")


def generate_teacher_outputs(cfg, logger):
    """
    Use Stable Diffusion x4 Upscaler to generate teacher outputs from LR.
    Skips files that already exist unless cfg.overwrite_data == True.
    If no files need generating, it won't even load the pipeline.
    """
    train_lr_paths = glob.glob(os.path.join(cfg.train_lr_folder, "*"))
    valid_lr_paths = glob.glob(os.path.join(cfg.valid_lr_folder, "*"))
    
    train_teacher_folder = cfg.train_teacher_folder
    valid_teacher_folder = cfg.valid_teacher_folder
    
    os.makedirs(train_teacher_folder, exist_ok=True)
    os.makedirs(valid_teacher_folder, exist_ok=True)
    
    # Determine which files actually need teacher outputs
    needed_train_paths = []
    for path in train_lr_paths:
        filename = os.path.basename(path)
        output_path = os.path.join(train_teacher_folder, filename)
        if cfg.overwrite_data or not os.path.exists(output_path):
            needed_train_paths.append(path)
    
    needed_valid_paths = []
    for path in valid_lr_paths:
        filename = os.path.basename(path)
        output_path = os.path.join(valid_teacher_folder, filename)
        if cfg.overwrite_data or not os.path.exists(output_path):
            needed_valid_paths.append(path)
    
    # If nothing needs to be generated, we can skip pipeline loading entirely
    total_needed = len(needed_train_paths) + len(needed_valid_paths)
    if total_needed == 0:
        logger.log("[Teacher]  All teacher outputs already exist, and overwrite_data=False. Skipping upscaling.")
        return
    
    # Otherwise, load pipeline
    logger.log("[Teacher]  Loading the teacher pipeline...")
    teacher_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        cfg.model_id, 
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(cfg.device)
    
    # Disable progress bar
    teacher_pipeline.set_progress_bar_config(disable=True)
    
    # --- Training teacher outputs ---
    if needed_train_paths:
        logger.log(f"[Teacher]  Generating teacher outputs for train LR: {len(needed_train_paths)} images.")
        for i, path in enumerate(needed_train_paths):
            filename = os.path.basename(path)
            output_path = os.path.join(train_teacher_folder, filename)
            
            lr_img = Image.open(path).convert("RGB")
            with torch.no_grad():
                upscaled = teacher_pipeline(
                    prompt=cfg.teacher_prompt,
                    image=lr_img,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale
                ).images[0]
            upscaled.save(output_path)
            
            if (i+1) % 50 == 0:
                logger.log(f"[Teacher]  Upscaling {i+1}/{len(needed_train_paths)} training images...")
    
    # --- Validation teacher outputs ---
    if needed_valid_paths:
        logger.log(f"[Teacher]  Generating teacher outputs for valid LR: {len(needed_valid_paths)} images.")
        for i, path in enumerate(needed_valid_paths):
            filename = os.path.basename(path)
            output_path = os.path.join(valid_teacher_folder, filename)
            
            lr_img = Image.open(path).convert("RGB")
            with torch.no_grad():
                upscaled = teacher_pipeline(
                    prompt=cfg.teacher_prompt,
                    image=lr_img,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale
                ).images[0]
            upscaled.save(output_path)
            
            if (i+1) % 10 == 0:
                logger.log(f"[Teacher]  Upscaling {i+1}/{len(needed_valid_paths)} validation images...")
