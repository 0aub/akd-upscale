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
    return f"{sum(param.numel() for param in model.parameters() if param.requires_grad):,}"

def collate_fn(batch):
    lr_img, teacher_img = batch[0]
    return lr_img, teacher_img

class Trainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        
        self.initialize_models()
        self.initializa_losses()
        self.initialize_dataloaders()
        self.initialize_optimizers()

    def initialize_models(self):
        # Student generator & Discriminator
        self.logger.log(f"\n[Trainer]  Initializing Student (G + D)...")
        self.generator = LightweightSRModel(
            up_factor=self.cfg.generator_up_factor,
            base_ch=self.cfg.generator_base_channels,
            num_blocks=self.cfg.generator_num_blocks
        ).to(self.cfg.device)

        self.discriminator = StrongPatchDiscriminator(
            in_ch=self.cfg.discriminator_in_channels, 
            base_ch=self.cfg.discriminator_base_channels
        ).to(self.cfg.device)

        self.logger.log(f"[Trainer]  G Total Parameters -> {get_total_trainable_parameters(self.generator)}")
        self.logger.log(f"[Trainer]  D Total Parameters -> {get_total_trainable_parameters(self.discriminator)}")

    def initializa_losses(self):
        # Loss function
        self.logger.log(f"\n[Trainer]  Initializing Loss Functions...")
        self.g_criterion = Loss(self.cfg.generator_loss, self.cfg.device).to(self.cfg.device)
        self.d_criterion = Loss(self.cfg.discriminator_loss, self.cfg.device).to(self.cfg.device)

    def initialize_dataloaders(self):
        # data preparation
        self.logger.log(f"\n[Trainer]  Initializing Dataloaders...")
        # Data transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

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
            num_workers=0,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    def initialize_optimizers(self):
        # Number of training steps per epoch        
        self.logger.log(f"\n[Trainer]  Initializing Optimizers...")
        num_train_steps = len(self.train_loader)

        self.g_optimizer = Optimizer(
            model_parameters=self.generator.parameters(),
            optimizer=self.cfg.generator_optimizer,
            learning_rate=self.cfg.generator_learning_rate,
            weight_decay=self.cfg.generator_weight_decay,
            betas=self.cfg.generator_betas,
            scheduler=self.cfg.generator_scheduler,
            warmup_steps=self.cfg.generator_warmup_steps,
            warmup_ratio=self.cfg.generator_warmup_ratio,
            num_train_steps=num_train_steps,
            epochs=self.cfg.epochs,
            clip_max_norm=self.cfg.generator_clip_max_norm
        )

        self.d_optimizer = Optimizer(
            model_parameters=self.discriminator.parameters(),
            optimizer=self.cfg.discriminator_optimizer,
            learning_rate=self.cfg.discriminator_learning_rate,
            weight_decay=self.cfg.discriminator_weight_decay,
            betas=self.cfg.discriminator_betas,
            scheduler=self.cfg.discriminator_scheduler,
            warmup_steps=self.cfg.discriminator_warmup_steps,
            warmup_ratio=self.cfg.discriminator_warmup_ratio,
            num_train_steps=num_train_steps,
            epochs=self.cfg.epochs,
            clip_max_norm=self.cfg.discriminator_clip_max_norm
        )

    def train(self):
        self.logger.log("\n[Trainer]  Starting Training...\n")
        start_time = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start_time = time.time()

            self._run_one_epoch(self.train_loader, training=True, epoch=epoch)
            self._run_one_epoch(self.valid_loader, training=False, epoch=epoch)

            epoch_duration = time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_start_time))
            self.logger.log_epoch(epoch, epoch_duration)

        total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        self.logger.log(f"\n[Trainer]  Training completed in {total_time}.\n")

    def _run_one_epoch(self, loader, training=True, epoch=1):
        mode = "Train" if training else "Valid"
        if training:
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()

        # Initialize metrics for the epoch
        self.logger.reset_metrics()
        time_elapsed = 0

        for step, (lr_img, teacher_img) in enumerate(loader, start=1):
            step_start_time = time.time()
            lr_img, teacher_img = lr_img.to(self.cfg.device), teacher_img.to(self.cfg.device)

            # Run training or validation step
            if training:
                self._train_step(lr_img, teacher_img)
            else:
                self._valid_step(lr_img, teacher_img)

            # Calculate step time
            step_time = time.time() - step_start_time
            time_elapsed += step_time
            self.logger.update_metrics('step_time', step_time)

            # Log at specified intervals
            if step % self.cfg.log_freq == 0 or step == len(loader):
                self.logger.log_step(epoch, step, len(loader), mode, time_elapsed, interval=True)
                self.logger.reset_metrics(interval=True)
                time_elapsed = 0

    def _train_step(self, lr_img, teacher_img):
        """Perform a single training step for both generator and discriminator."""
        # -------------------------------
        #       Discriminator update
        # -------------------------------
        self.d_optimizer.zero_grad()

        # Discriminator loss
        fake_img = self.generator(lr_img).detach()
        real_out = self.discriminator(teacher_img)
        fake_out = self.discriminator(fake_img.detach())

        target_real = torch.ones_like(real_out)
        target_fake = torch.zeros_like(fake_out)

        d_loss_real = self.d_criterion(real_out, target_real).mean()
        d_loss_fake = self.d_criterion(fake_out, target_fake).mean()
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        # Backpropagation
        d_loss.backward()
        d_grad_norm = self.d_optimizer.clip_grad_norm(self.discriminator.parameters())
        self.d_optimizer.step()

        # -------------------------------
        #       Generator update
        # -------------------------------
        self.g_optimizer.zero_grad()

        # Generator loss
        fake_img = self.generator(lr_img)
        fake_out_for_g = self.discriminator(fake_img)

        target_gen = torch.ones_like(fake_out_for_g)

        g_loss_content = self.g_criterion(fake_img, teacher_img).mean()
        g_loss_adv = self.cfg.adversarial_weight * self.d_criterion(fake_out_for_g, target_gen).mean()
        g_loss = g_loss_content + g_loss_adv

        # Backpropagation
        g_loss.backward()
        g_grad_norm = self.g_optimizer.clip_grad_norm(self.generator.parameters())
        self.g_optimizer.step()

        # -------------------------------
        #       Logging
        # -------------------------------
        self.logger.update_metrics("g_grad_norm", g_grad_norm)
        self.logger.update_metrics("d_grad_norm", d_grad_norm)

        self.logger.update_metrics('g_lr', self.g_optimizer.get_lr()) 
        self.logger.update_metrics('d_lr', self.d_optimizer.get_lr())

        self.logger.update_metrics('g_loss', g_loss.item())
        self.logger.update_metrics('d_loss', d_loss.item())


    def _valid_step(self, lr_img, teacher_img):
        """Perform a single validation step for generator."""
        with torch.no_grad():
            # Generator loss
            fake_img = self.generator(lr_img)
            fake_out_for_g = self.discriminator(fake_img)

            target_gen = torch.ones_like(fake_out_for_g)

            g_loss_content = self.g_criterion(fake_img, teacher_img).mean()
            g_loss_adv = self.cfg.adversarial_weight * self.d_criterion(fake_out_for_g, target_gen).mean()
            g_loss = g_loss_content + g_loss_adv
        self.logger.update_metrics('g_loss', g_loss.item())


    def test(self):
        result_folder = os.path.join(self.logger.exp_path, "results")
        os.makedirs(result_folder, exist_ok=True)

        # Evaluate on the low_resolution_folder/test + teacher_folder/test
        test_lr_dir = os.path.join(self.cfg.low_resolution_folder, "test")
        test_teacher_dir = os.path.join(self.cfg.teacher_folder, "test")

        test_files = sorted(glob.glob(os.path.join(test_lr_dir, "*")))
        self.logger.log(f"\n[Trainer]  Found {len(test_files)} test LR images.\n")

        self.generator.eval()
        self.discriminator.eval()

        time_since_last_print = 0
        cumulative_test_time = 0

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
                time_since_last_print += elapsed
                cumulative_test_time += elapsed

                if idx % self.cfg.log_freq == 0 or idx == len(test_files):
                    avg_step_time = time_since_last_print / self.cfg.log_freq
                    self.logger.log(
                        f"[Trainer]  Test {idx}/{len(test_files)} => "
                        f"Time: {time_since_last_print:.2f}s (Avg: {avg_step_time:.2f}s/Step)"
                    )
                    time_since_last_print  = 0  

        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - cumulative_test_time))
        self.logger.log(f"\n[Trainer]  Testing completed in {total_time_str}")

    def save_model(self, path):
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }, path)
        self.logger.log(f"[Trainer]  Model (G + D) saved to: {path}")
