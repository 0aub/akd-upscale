import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class UpscaleDataset(Dataset):
    """
    Returns (LR, Teacher) pairs.
    """
    def __init__(self, lr_folder, teacher_folder, transform=None):
        super().__init__()
        self.lr_paths = sorted(glob.glob(os.path.join(lr_folder, "*")))
        self.teacher_paths = sorted(glob.glob(os.path.join(teacher_folder, "*")))
        self.transform = transform
        
        if len(self.lr_paths) != len(self.teacher_paths):
            print(f"[Warning] Mismatch: {len(self.lr_paths)} LR images, {len(self.teacher_paths)} teacher images.")
    
    def __len__(self):
        return len(self.lr_paths)
    
    def __getitem__(self, idx):
        lr_path = self.lr_paths[idx]
        teacher_path = self.teacher_paths[idx]
        
        lr_img = Image.open(lr_path).convert("RGB")
        teacher_img = Image.open(teacher_path).convert("RGB")
        
        if self.transform:
            lr_img = self.transform(lr_img)
            teacher_img = self.transform(teacher_img)
        
        return lr_img, teacher_img
