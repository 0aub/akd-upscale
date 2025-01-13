import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import UpscaleDataset
from .model import TinyUpscaler
from .optim import Optimizer

class Trainer:
    """
    Orchestrates the training of the student model 
    by comparing outputs to the teacher outputs.
    """
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        
        # Student model
        self.model = TinyUpscaler(up_factor=cfg["up_factor"]).to(cfg["device"])
        
        # Loss function
        self.criterion = nn.L1Loss()
        
        # Data transforms
        self.transform = transforms.ToTensor()
        
        # Datasets
        self.train_dataset = UpscaleDataset(
            lr_folder=cfg["train_lr_folder"],
            teacher_folder=cfg["train_teacher_folder"],
            transform=self.transform
        )
        self.valid_dataset = UpscaleDataset(
            lr_folder=cfg["valid_lr_folder"],
            teacher_folder=cfg["valid_teacher_folder"],
            transform=self.transform
        )
        
        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=2
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=2
        )
        
        # Number of training steps per epoch
        self.num_train_steps = len(self.train_loader)
        
        self.optimizer = Optimizer(
            model_parameters=self.model.parameters(),
            config=cfg,
            num_train_steps=self.num_train_steps
        )
    
    def train(self):
        self.logger.log("[Trainer] Starting knowledge distillation training...")
        for epoch in range(self.cfg["epochs"]):
            train_loss = self._run_one_epoch(self.train_loader, training=True)
            valid_loss = self._run_one_epoch(self.valid_loader, training=False)
            
            self.logger.log(
                f"Epoch {epoch+1}/{self.cfg['epochs']}  ->  Train Loss: {train_loss:.4f}  |  Valid Loss: {valid_loss:.4f}  |  Grad Norm: {self.optimizer.clip_grad_norm(self.model.parameters()):.6f}  |  LR: {self.optimizer.get_lr():.6f}"
            )
            self.logger.logline()
    
    def _run_one_epoch(self, loader, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0.0
        for step, (lr_batch, teacher_batch) in enumerate(loader):
            lr_batch = lr_batch.to(self.cfg["device"])
            teacher_batch = teacher_batch.to(self.cfg["device"])
            
            if training:
                self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(training):
                out = self.model(lr_batch)
                loss = self.criterion(out, teacher_batch)
                
                if training:
                    loss.backward()
                    # Gradient clipping
                    self.optimizer.clip_grad_norm(self.model.parameters())
                    self.optimizer.step(current_step=step)
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.log(f"[Trainer] Model saved to: {path}")
