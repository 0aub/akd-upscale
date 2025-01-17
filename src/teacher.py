import os
import glob
import math
import shutil 
from PIL import Image
import torch
from diffusers import StableDiffusionUpscalePipeline


class TeacherUpscaler:
    """
    1) Takes a single LR folder (cfg.lr_folder).
    2) Splits:
       - first 100 => test
       - remainder => 80/20 => train/val.
    3) Upscales each split with the teacher pipeline:
       - saves in teacher_folder/<split>
       - also moves or copies LR images to lr_folder/<split>
         so the Trainer can pick them up easily.
    """

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.teacher_pipeline = None

        # Make subfolders for LR
        self.lr_train_dir = os.path.join(self.cfg.lr_folder, "train")
        self.lr_val_dir   = os.path.join(self.cfg.lr_folder, "val")
        self.lr_test_dir  = os.path.join(self.cfg.lr_folder, "test")

        # Make subfolders for teacher
        self.teacher_train_dir = os.path.join(self.cfg.teacher_folder, "train")
        self.teacher_val_dir   = os.path.join(self.cfg.teacher_folder, "val")
        self.teacher_test_dir  = os.path.join(self.cfg.teacher_folder, "test")

        # Create directories
        for d in [self.lr_train_dir, self.lr_val_dir, self.lr_test_dir,
                  self.teacher_train_dir, self.teacher_val_dir, self.teacher_test_dir]:
            os.makedirs(d, exist_ok=True)

    def load_teacher_pipeline(self):
        if self.teacher_pipeline is None:
            self.logger.log("[Teacher]  Loading StableDiffusionUpscalePipeline...")
            self.teacher_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                self.cfg.model_id,
                torch_dtype=torch.float32,
                safety_checker=None
            ).to(self.cfg.device)
            self.teacher_pipeline.set_progress_bar_config(disable=True)
        return self.teacher_pipeline

    def split_data(self):
        """
        Splits the images in self.cfg.lr_folder (the "root" LR) as follows:
          - first 100 => test
          - remainder => 80/20 => train/val
        Returns 3 lists: (test_list, train_list, val_list)
        """
        all_files = sorted(
            f for f in glob.glob(os.path.join(self.cfg.lr_folder, "*"))
            if os.path.isfile(f)  # exclude subdirs
        )

        # Exclude subdirs like lr_folder/train if they already exist from a previous run
        # We'll only split the top-level images.  If you want to handle that differently,
        # adapt as needed.
        # For safety, let's keep only those that are in the root folder (no slash).
        root_files = [f for f in all_files if os.path.dirname(f) == self.cfg.lr_folder]

        if len(root_files) == 0:
            self.logger.log("[Teacher]  No top-level LR images found. Possibly already split?")
            return [], [], []

        num_test = min(100, len(root_files))
        test_list = root_files[:num_test]
        remainder = root_files[num_test:]
        split_idx = math.floor(0.8 * len(remainder))
        train_list = remainder[:split_idx]
        val_list   = remainder[split_idx:]

        self.logger.log(
            f"\n[Teacher]  Dataset Summary\n"
            f"{'-'*40}\n"
            f"Total LR Images Found: {len(root_files):>5}\n"
            f"  - Training Images:    {len(train_list):>5}\n"
            f"  - Validation Images:  {len(val_list):>5}\n"
            f"  - Test Images:        {len(test_list):>5}\n"
            f"{'-'*40}"
        )
        return test_list, train_list, val_list

    def move_or_copy(self, src_path, dst_folder):
        """
        Moves (or copies) a file from src_path -> dst_folder.
        Here we do a 'move'. If you'd rather copy, use shutil.copy2.
        """
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_folder, filename)

        # If we do not want to overwrite & file exists, skip
        if os.path.exists(dst_path) and not self.cfg.overwrite_teacher_data:
            return
        # move
        shutil.move(src_path, dst_path)

    def run(self):
        test_list, train_list, val_list = self.split_data()

        # Move LR images into subfolders
        for f in test_list:
            self.move_or_copy(f, self.lr_test_dir)
        for f in train_list:
            self.move_or_copy(f, self.lr_train_dir)
        for f in val_list:
            self.move_or_copy(f, self.lr_val_dir)

        # Now we have:
        #  low_resolution_folder/train/*.png
        #  low_resolution_folder/val/*.png
        #  low_resolution_folder/test/*.png

        # Upscale each split with teacher
        self.upscale_split(self.lr_train_dir, self.teacher_train_dir, "train")
        self.upscale_split(self.lr_val_dir,   self.teacher_val_dir,   "val")
        self.upscale_split(self.lr_test_dir,  self.teacher_test_dir,  "test")

        self.logger.log("[Teacher]  Done splitting and upscaling.")

    def upscale_split(self, lr_dir, teacher_dir, split_name):
        """
        Upscale all images in lr_dir, save them to teacher_dir.
        """
        lr_files = sorted(glob.glob(os.path.join(lr_dir, "*")))
        if not lr_files:
            self.logger.log(f"[Teacher]  No LR images found in {lr_dir}, skip {split_name} upscaling.")
            return
        
        pipe = self.load_teacher_pipeline()
        self.logger.log(f"[Teacher]  Upscaling {len(lr_files)} {split_name} images => {teacher_dir}")

        for i, path in enumerate(lr_files, start=1):
            basename = os.path.basename(path)
            out_path = os.path.join(teacher_dir, basename)

            if os.path.exists(out_path) and not self.cfg.overwrite_teacher_data:
                continue

            lr_img = Image.open(path).convert("RGB")
            with torch.no_grad():
                out_img = pipe(
                    prompt=self.cfg.teacher_prompt,
                    image=lr_img,
                    num_inference_steps=self.cfg.num_inference_steps,
                    guidance_scale=self.cfg.guidance_scale
                ).images[0]
            out_img.save(out_path)

            if i % 50 == 0:
                self.logger.log(f"[Teacher]  {split_name} => {i}/{len(lr_files)} upscaled.")

