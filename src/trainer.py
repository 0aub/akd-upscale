import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import time

from .dataset import UpscaleDataset
from .model import TinyUpscaler
from .optim import Optimizer
from .teacher import DownscaleByFactor

def collate_fn(batch):
    """
    For batch_size=1, 'batch' is a list of length 1, like:
        [ (lr_image, teacher_image) ]
    We just return that single pair.
    """
    lr_img, teacher_img = batch[0]
    return lr_img, teacher_img


class Trainer:
    """
    Orchestrates the training of the student model 
    by comparing outputs to the teacher outputs.
    """
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        
        # Student model
        self.model = TinyUpscaler(up_factor=cfg.up_factor).to(cfg.device)
        
        # Loss function
        self.criterion = nn.L1Loss()
        
        # Data transforms
        self.transform = transforms.Compose([
            DownscaleByFactor(cfg.up_factor),
            transforms.ToTensor(),
        ])

        # Datasets
        self.train_dataset = UpscaleDataset(
            lr_folder=cfg.train_lr_folder,
            teacher_folder=cfg.train_teacher_folder,
            transform=self.transform
        )
        self.valid_dataset = UpscaleDataset(
            lr_folder=cfg.valid_lr_folder,
            teacher_folder=cfg.valid_teacher_folder,
            transform=self.transform
        )
        
        # Dataloaders (batch_size=1)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        # Number of training steps per epoch
        self.num_train_steps = len(self.train_loader)
        
        self.optimizer = Optimizer(
            model_parameters=self.model.parameters(),
            config=cfg,
            num_train_steps=self.num_train_steps
        )
    
    def train(self):
        self.logger.log("[Trainer]  Starting knowledge distillation training...\n")
        for epoch in range(self.cfg.epochs):
            # Start timer for this epoch
            epoch_start_time = time.time()

            # Run training + validation
            train_loss = self._run_one_epoch(self.train_loader, training=True)
            valid_loss = self._run_one_epoch(self.valid_loader, training=False)
            
            # Clip gradients and measure grad norm
            grad_norm = self.optimizer.clip_grad_norm(self.model.parameters())

            # Compute elapsed time for this epoch
            epoch_time = time.time() - epoch_start_time

            # Log everything, including the time in seconds
            self.logger.log(
                f"Epoch {epoch+1}/{self.cfg.epochs}  ->  "
                f"Train Loss: {train_loss:.4f}  |  Valid Loss: {valid_loss:.4f}  |  "
                f"Grad Norm: {grad_norm:.6f}  |  LR: {self.optimizer.get_lr():.6f}  |  "
                f"Time: {epoch_time:.2f}s"
            )
            
    
    def _run_one_epoch(self, loader, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0.0
        for step, (lr_img, teacher_img) in enumerate(loader):
            # Both are single images, shape: [3, 256, 256]
            lr_img = lr_img.to(self.cfg.device)
            teacher_img = teacher_img.to(self.cfg.device)

            # For the model, we typically want shape [B, C, H, W]
            # For batch_size=1, unsqueeze(0) adds the batch dimension.
            lr_img = lr_img.unsqueeze(0)       # [1, 3, 256, 256]
            teacher_img = teacher_img.unsqueeze(0)  # [1, 3, 256, 256]
            
            if training:
                self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(training):
                out = self.model(lr_img)  # out -> [1, 3, 256, 256]
                loss = self.criterion(out, teacher_img)
                
                if training:
                    loss.backward()
                    self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.log(f"[Trainer] Model saved to: {path}")
