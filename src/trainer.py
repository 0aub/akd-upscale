import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import time
import glob
import os
from PIL import Image


from .dataset import UpscaleDataset
from .model import LightweightSRModel, StrongPatchDiscriminator
from .optim import Optimizer
from .teacher import DownscaleByFactor, upscale_image
from .loss import Loss

def collate_fn(batch):
    lr_img, teacher_img = batch[0]
    return lr_img, teacher_img


class Trainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        
        # Student models
        self.generator = LightweightSRModel(
            up_factor=cfg.up_factor,
            base_ch=32,
            num_blocks=4
        ).to(cfg.device)
        
        self.discriminator = StrongPatchDiscriminator(
            in_ch=3, 
            base_ch=64
        ).to(cfg.device)
        
        # Loss function
        self.g_criterion = Loss(cfg.generator_loss, cfg.device).to(cfg.device)
        self.d_criterion = Loss(cfg.discriminator_loss, cfg.device).to(cfg.device)
        
        # data preparation
        self.prepare_dataloaders()

        # Number of training steps per epoch
        num_train_steps = len(self.train_loader)
        
        self.g_optimizer = Optimizer(
            model_parameters=self.generator.parameters(),
            optimizer=cfg.generator_optimizer,
            learning_rate=cfg.generator_learning_rate,
            weight_decay=cfg.generator_weight_decay,
            betas=cfg.generator_betas,
            scheduler=cfg.generator_scheduler,
            warmup_steps=cfg.generator_warmup_steps,
            warmup_ratio=cfg.generator_warmup_ratio,
            num_train_steps=num_train_steps,
            epochs=cfg.epochs,
            clip_max_norm=cfg.generator_clip_max_norm
        )

        self.d_optimizer = Optimizer(
            model_parameters=self.discriminator.parameters(),
            optimizer=cfg.discriminator_optimizer,
            learning_rate=cfg.discriminator_learning_rate,
            weight_decay=cfg.discriminator_weight_decay,
            betas=cfg.discriminator_betas,
            scheduler=cfg.discriminator_scheduler,
            warmup_steps=cfg.discriminator_warmup_steps,
            warmup_ratio=cfg.discriminator_warmup_ratio,
            num_train_steps=num_train_steps,
            epochs=cfg.epochs,
            clip_max_norm=cfg.discriminator_clip_max_norm
        )
    
    def prepare_dataloaders(self):
        # Data transforms
        self.transform = transforms.Compose([
            DownscaleByFactor(self.cfg.up_factor),
            transforms.ToTensor(),
        ])

        # Datasets
        self.train_dataset = UpscaleDataset(
            lr_folder=self.cfg.train_lr_folder,
            teacher_folder=self.cfg.train_teacher_folder,
            transform=self.transform
        )
        self.valid_dataset = UpscaleDataset(
            lr_folder=self.cfg.valid_lr_folder,
            teacher_folder=self.cfg.valid_teacher_folder,
            transform=self.transform
        )
        
        # Dataloaders (batch_size=1)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )

    def train(self):
        """
        Runs the main training loop across all epochs,
        computing both G and D losses and logging them.
        """
        self.logger.log("[Trainer] Starting Training...\n")
        start_time = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start_time = time.time()

            # Run one epoch of training
            train_g_loss, train_d_loss = self._run_one_epoch(self.train_loader, training=True)
            # Run one epoch of validation (no parameter updates)
            val_g_loss, val_d_loss = self._run_one_epoch(self.valid_loader, training=False)

            # Clip gradient norms (if desired) after the epoch
            g_grad_norm = self.g_optimizer.clip_grad_norm(self.generator.parameters())
            d_grad_norm = self.d_optimizer.clip_grad_norm(self.discriminator.parameters())

            # Get current learning rates
            g_lr = self.g_optimizer.get_lr()
            d_lr = self.d_optimizer.get_lr()

            # Compute epoch time
            epoch_time = time.time() - epoch_start_time

            # Log results
            self.logger.log(
                f"[Epoch {epoch}/{self.cfg.epochs}] "
                f"Train Loss | G: {train_g_loss:.4f}, D: {train_d_loss:.4f} || "
                f"Val Loss | G: {val_g_loss:.4f}, D: {val_d_loss:.4f} || "
                f"Grad Norm | G: {g_grad_norm:.6f}, D: {d_grad_norm:.6f} || "
                f"LR | G: {g_lr:.6f}, D: {d_lr:.6f} || "
                f"Time: {epoch_time:.2f}s"
            )


        total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        self.logger.log(f"\n[Trainer] Training completed in {total_time}.\n")


    def _run_one_epoch(self, loader, training=True):
        """
        Runs a single epoch of training or validation.
        For training=True:
        1) Updates Discriminator (real->1, fake->0).
        2) Updates Generator (reconstruction/distillation + adversarial).
        For training=False:
        Simply measures losses without optimizer steps.
        """
        if training:
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for lr_img, teacher_img in loader:
            # Move data to device
            lr_img = lr_img.to(self.cfg.device).unsqueeze(0)       # [B=1,3,H,W]
            teacher_img = teacher_img.to(self.cfg.device).unsqueeze(0)

            ###########################################################
            # 1) Update Discriminator (only in training)
            ###########################################################
            if training:
                # Zero out gradients for D
                self.d_optimizer.zero_grad()
                
                # Generate fake image without gradient for G
                with torch.no_grad():
                    fake_img = self.generator(lr_img)  # shape: [1,3,H,W]
                
                # ----- Real pass -----
                real_out = self.discriminator(teacher_img)
                real_label = torch.ones_like(real_out, device=self.cfg.device)
                d_loss_real = self.d_criterion(real_out, real_label)

                # ----- Fake pass -----
                fake_out = self.discriminator(fake_img.detach())
                fake_label = torch.zeros_like(fake_out, device=self.cfg.device)
                d_loss_fake = self.d_criterion(fake_out, fake_label)

                # Combine D losses
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_loss.backward()
                self.d_optimizer.step()
            else:
                # Validation mode => no D updates
                d_loss = torch.tensor(0.0, device=self.cfg.device)

            ###########################################################
            # 2) Update Generator
            ###########################################################
            if training:
                self.g_optimizer.zero_grad()
            
            # Forward pass for G
            gen_out = self.generator(lr_img)

            # (a) Reconstruction/distillation loss
            recon_loss = self.g_criterion(gen_out, teacher_img)

            # (b) Adversarial loss => want D(gen_out)=1
            fake_out_for_g = self.discriminator(gen_out)
            adv_target = torch.ones_like(fake_out_for_g, device=self.cfg.device)
            adv_loss = self.d_criterion(fake_out_for_g, adv_target)

            # Combine the generator losses
            # Example weighting: self.cfg.adv_weight 
            g_loss = recon_loss + (self.cfg.adv_weight * adv_loss)

            if training:
                g_loss.backward()
                self.g_optimizer.step()
            else:
                # No backward step in validation
                pass

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        # Average G and D losses across the dataset
        avg_g_loss = total_g_loss / len(loader)
        avg_d_loss = total_d_loss / len(loader)

        return avg_g_loss, avg_d_loss
    
    def test(self):
        test_folder = self.cfg.test_path
        result_folder = os.path.join(self.logger.exp_path, "results")
        os.makedirs(result_folder, exist_ok=True)
        
        # Switch both G and D to eval (though D isn't strictly needed here)
        self.generator.eval()
        self.discriminator.eval()
        
        # Gather test files
        test_files = glob.glob(os.path.join(test_folder, "*"))
        self.logger.log(f"[Trainer] Found {len(test_files)} test images in: {test_folder}")
        
        max_test_count = min(self.cfg.test_count, len(test_files))
        test_files = test_files[:max_test_count]
        
        test_start_time = time.time()
        
        with torch.no_grad():
            for i, fpath in enumerate(test_files, start=1):
                img_start_time = time.time()

                # 1) Load the image as PIL
                img = Image.open(fpath).convert("RGB")

                # 2) Student Upscale
                #    (Apply the same transform pipeline => produce a tensor => feed generator)
                inp_tensor = self.transform(img)  # shape [3,H,W]
                inp_tensor = inp_tensor.unsqueeze(0).to(self.cfg.device)  # [1,3,H,W]
                out_student = self.generator(inp_tensor)
                out_student = out_student.squeeze(0).cpu().clamp(0,1)  # [3,H*,W*]
                student_upscaled_pil = transforms.ToPILImage()(out_student)

                # 3) Teacher Upscale
                #    (Use the teacher's "upscale_image()" function, which takes the LR PIL)
                teacher_upscaled_pil = upscale_image(self.cfg, img)  # returns a PIL image

                # 4) Plot LR | Student | Teacher side by side
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                # Left: Low-res (original input)
                axs[0].imshow(img)
                axs[0].set_title("LR Input")
                axs[0].axis("off")

                # Middle: Student
                axs[1].imshow(student_upscaled_pil)
                axs[1].set_title("Student Upscaled")
                axs[1].axis("off")

                # Right: Teacher
                axs[2].imshow(teacher_upscaled_pil)
                axs[2].set_title("Teacher Upscaled")
                axs[2].axis("off")

                # Adjust layout & save figure
                plt.tight_layout()
                basename = os.path.basename(fpath)
                save_name = os.path.splitext(basename)[0] + "_comparison.png"
                save_path = os.path.join(result_folder, save_name)
                plt.savefig(save_path, dpi=150)
                plt.close(fig)

                # Log timing
                img_time = time.time() - img_start_time
                self.logger.log(
                    f"[Trainer] Upscaling {i}/{max_test_count} -> {save_name}  (Time: {img_time:.2f}s)"
                )

        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - test_start_time))
        self.logger.log(f"\n[Trainer] Test images saved to: {result_folder}")
        self.logger.log(f"[Trainer] Testing completed in {total_time_str}")

    def save_model(self, path):
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
            },
            path
        )
        self.logger.log(f"[Trainer] Model (G + D) saved to: {path}")
