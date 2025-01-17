import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import time
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt

from .dataset import UpscaleDataset
from .model import LightweightSRModel, StrongPatchDiscriminator
from .optim import Optimizer
from .loss import Loss


def get_total_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def collate_fn(batch):
    lr_img, teacher_img = batch[0]
    return lr_img, teacher_img


class Trainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        
        # Student generator & Discriminator
        self.generator = LightweightSRModel(
            up_factor=cfg.generator_up_factor,
            base_ch=cfg.generator_base_channels,
            num_blocks=cfg.generator_num_blocks
        ).to(cfg.device)

        self.discriminator = StrongPatchDiscriminator(
            in_ch=cfg.discriminator_in_channels, 
            base_ch=cfg.discriminator_base_channels
        ).to(cfg.device)

        logger.log(f"G Total Parameters -> {get_total_trainable_parameters(self.generator)}")
        logger.log(f"D Total Parameters -> {get_total_trainable_parameters(self.discriminator)}")
        
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
        self.transform = transforms.ToTensor()

        # Datasets
        train_lr_dir = os.path.join(self.cfg.low_resolution_folder, "train")
        train_teacher_dir = os.path.join(self.cfg.teacher_folder, "train")
        self.train_dataset = UpscaleDataset(train_lr_dir, train_teacher_dir, transform=transform)

        val_lr_dir = os.path.join(self.cfg.low_resolution_folder, "val")
        val_teacher_dir = os.path.join(self.cfg.teacher_folder, "val")
        self.valid_dataset = UpscaleDataset(val_lr_dir, val_teacher_dir, transform=transform)

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
        self.logger.log("[Trainer]  Starting Training...\n")
        start_time = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start_time = time.time()

            # Run one epoch of training
            train_g_loss, train_d_loss = self._run_one_epoch(self.train_loader, training=True, epoch=epoch)
            # Run one epoch of validation (no parameter updates)
            val_g_loss, val_d_loss = self._run_one_epoch(self.valid_loader, training=False, epoch=epoch)

            # Clip gradient norms (if desired) after the epoch
            g_grad_norm = self.g_optimizer.clip_grad_norm(self.generator.parameters())
            d_grad_norm = self.d_optimizer.clip_grad_norm(self.discriminator.parameters())

            # Get current learning rates
            g_lr = self.g_optimizer.get_lr()
            d_lr = self.d_optimizer.get_lr()

            # Compute epoch time
            epoch_time = time.time() - epoch_start_time

            # Log results for the epoch
            self.logger.log(
                f"\n{'='*50}\n"
                f"[Epoch {epoch}/{self.cfg.epochs}] Summary:\n"
                f"\tTrain Loss | G: {train_g_loss:.4f}  , D: {train_d_loss:.4f}\n"
                f"\tVal Loss   | G: {val_g_loss:.4f}  , D: {val_d_loss:.4f}\n"
                f"\tGrad Norm  | G: {g_grad_norm:.6f}, D: {d_grad_norm:.6f}\n"
                f"\tLR         | G: {g_lr:.6f}, D: {d_lr:.6f}\n"
                f"\tTime       | {epoch_time:.2f}s\n"
                f"{'='*50}\n"
            )

        total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        self.logger.log(f"\n[Trainer]  Training completed in {total_time}.\n")



    def _run_one_epoch(self, loader, training=True, epoch=1):
        if training:
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for step, (lr_img, teacher_img) in enumerate(loader, start=1):
            step_start_time = time.time()

            # Move data to device
            lr_img = lr_img.to(self.cfg.device).unsqueeze(0)       # [B=1,3,H,W]
            teacher_img = teacher_img.to(self.cfg.device).unsqueeze(0)

            # ---------------------------------------------------------
            # 1) Update Discriminator
            # ---------------------------------------------------------
            if training:
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

            # ---------------------------------------------------------
            # 2) Update Generator
            # ---------------------------------------------------------
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

            # Step-wise logging
            step_time = time.time() - step_start_time
            if training:
                g_lr = self.g_optimizer.get_lr()
                d_lr = self.d_optimizer.get_lr()
                g_grad_norm = self.g_optimizer.clip_grad_norm(self.generator.parameters())
                d_grad_norm = self.d_optimizer.clip_grad_norm(self.discriminator.parameters())

                self.logger.log(
                    f"[Epoch {epoch} | Train Step {step}/{len(loader)}] "
                    f"Loss | G: {g_loss.item():.4f}, D: {d_loss.item():.4f} || "
                    f"Grad Norm | G: {g_grad_norm:.6f}, D: {d_grad_norm:.6f} || "
                    f"LR | G: {g_lr:.6f}, D: {d_lr:.6f} || "
                    f"Time: {step_time:.2f}s"
                )
            else:
                self.logger.log(
                    f"[Epoch {epoch} | Valid Step {step}/{len(loader)}] "
                    f"Loss | G: {g_loss.item():.4f}, D: {d_loss.item():.4f} || "
                    f"Time: {step_time:.2f}s"
                )


        # Average G and D losses across the dataset
        avg_g_loss = total_g_loss / len(loader)
        avg_d_loss = total_d_loss / len(loader)

        return avg_g_loss, avg_d_loss

    def test(self):
        result_folder = os.path.join(self.logger.exp_path, "results")
        os.makedirs(result_folder, exist_ok=True)

        # Evaluate on the low_resolution_folder/test + teacher_folder/test
        test_lr_dir = os.path.join(self.cfg.low_resolution_folder, "test")
        test_teacher_dir = os.path.join(self.cfg.teacher_folder, "test")

        test_files = sorted(glob.glob(os.path.join(test_lr_dir, "*")))
        self.logger.log(f"[Trainer] Found {len(test_files)} test LR images.")

        self.generator.eval()
        self.discriminator.eval()

        start_time = time.time()

        with torch.no_grad():
            for idx, fpath in enumerate(test_files, start=1):
                img_start = time.time()
                filename = os.path.basename(fpath)

                # LR
                lr_pil = Image.open(fpath).convert("RGB")
                lr_tensor = transforms.ToTensor()(lr_pil).unsqueeze(0).to(self.cfg.device)
                student_out = self.generator(lr_tensor).squeeze(0).cpu().clamp(0,1)
                student_pil = transforms.ToPILImage()(student_out)

                # Teacher
                teacher_path = os.path.join(test_teacher_dir, filename)
                if os.path.exists(teacher_path):
                    teacher_pil = Image.open(teacher_path).convert("RGB")
                else:
                    # If teacher doesn't exist, skip or handle
                    teacher_pil = None

                # Plot side-by-side
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(lr_pil)
                axs[0].set_title("LR")
                axs[0].axis("off")

                axs[1].imshow(student_pil)
                axs[1].set_title("Student")
                axs[1].axis("off")

                if teacher_pil:
                    axs[2].imshow(teacher_pil)
                    axs[2].set_title("Teacher")
                else:
                    axs[2].imshow(student_pil)
                    axs[2].set_title("Teacher Missing")
                axs[2].axis("off")

                plt.tight_layout()
                out_name = os.path.splitext(filename)[0] + "_comparison.png"
                out_path = os.path.join(result_folder, out_name)
                plt.savefig(out_path, dpi=150)
                plt.close(fig)

                elapsed = time.time() - img_start
                self.logger.log(f"[Trainer] Test {idx}/{len(test_files)} => {out_name} (Time: {elapsed:.2f}s)")

        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        self.logger.log(f"[Trainer] Testing completed in {total_time_str}")

    def save_model(self, path):
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }, path)
        self.logger.log(f"[Trainer] Model (G + D) saved to: {path}")
