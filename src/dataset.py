import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class UpscaleDataset(Dataset):
    """
    Returns (LR, HR) pairs.
    """
    def __init__(self, lr_folder, hr_folder, transform=None):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.lr_images = sorted(os.listdir(lr_folder))
        self.hr_images = sorted(os.listdir(hr_folder))
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_folder, self.lr_images[idx])
        hr_path = os.path.join(self.hr_folder, self.hr_images[idx])

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img
