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
    Downscale HR images by cfg['up_factor'] to produce LR images.
    """
    train_hr_folder = cfg.train_hr_folder
    valid_hr_folder = cfg.valid_hr_folder
    
    train_lr_folder = cfg.train_lr_folder
    valid_lr_folder = cfg.valid_lr_folder
    
    os.makedirs(train_lr_folder, exist_ok=True)
    os.makedirs(valid_lr_folder, exist_ok=True)
    
    # Define a custom transform pipeline
    downscale_transform = transforms.Compose([
        DownscaleByFactor(cfg.up_factor)
    ])
    
    # Training HR -> LR
    train_hr_paths = glob.glob(os.path.join(train_hr_folder, "*"))
    logger.log(f"[Teacher]  Generating LR from training HR: {train_hr_folder}, found {len(train_hr_paths)} images.")
    
    for i, path in enumerate(train_hr_paths):
        img = Image.open(path).convert("RGB")
        lr_img = downscale_transform(img)
        filename = os.path.basename(path)
        lr_img.save(os.path.join(train_lr_folder, filename))
        if (i+1) % 50 == 0:
            logger.log(f"[Teacher]  Generating LR {i+1}/{len(train_hr_paths)}")
    
    # Validation HR -> LR
    valid_hr_paths = glob.glob(os.path.join(valid_hr_folder, "*"))
    logger.log(f"[Teacher] Generating LR from validation HR: {valid_hr_folder}, found {len(valid_hr_paths)} images.")
    
    for i, path in enumerate(valid_hr_paths):
        img = Image.open(path).convert("RGB")
        lr_img = downscale_transform(img)
        filename = os.path.basename(path)
        lr_img.save(os.path.join(valid_lr_folder, filename))
        if (i+1) % 50 == 0:
            logger.log(f"[Teacher]  Generating LR {i+1}/{len(train_hr_paths)}")

def generate_teacher_outputs(cfg, logger):
    """
    Use Stable Diffusion x4 Upscaler to generate teacher outputs from LR.
    """
    
    logger.log("[Teacher] Loading the teacher pipeline...")
    teacher_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        cfg.model_id, 
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(cfg.teacher_device)

    teacher_pipeline.set_progress_bar_config(disable=True)
    
    train_lr_paths = glob.glob(os.path.join(cfg.train_lr_folder, "*"))
    valid_lr_paths = glob.glob(os.path.join(cfg.valid_lr_folder, "*"))
    
    train_teacher_folder = cfg.train_teacher_folder
    valid_teacher_folder = cfg.valid_teacher_folder
    
    os.makedirs(train_teacher_folder, exist_ok=True)
    os.makedirs(valid_teacher_folder, exist_ok=True)
    
    # Training teacher outputs
    logger.log(f"[Teacher] Generating teacher outputs for train LR: {len(train_lr_paths)} images.")
    for i, path in enumerate(train_lr_paths):
        filename = os.path.basename(path)
        output_path = os.path.join(train_teacher_folder, filename)
        
        if os.path.exists(output_path):
            continue
        
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
            logger.log(f"[Teacher] Upscaling {i+1}/{len(train_lr_paths)} training images...")
    
    # Validation teacher outputs
    logger.log(f"[Teacher] Generating teacher outputs for valid LR: {len(valid_lr_paths)} images.")
    for i, path in enumerate(valid_lr_paths):
        filename = os.path.basename(path)
        output_path = os.path.join(valid_teacher_folder, filename)
        
        if os.path.exists(output_path):
            continue
        
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
            logger.log(f"[Teacher] Upscaling {i+1}/{len(valid_lr_paths)} validation images...")
