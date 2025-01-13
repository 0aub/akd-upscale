import torch.nn as nn
import torch.nn.functional as F

class TinyUpscaler(nn.Module):
    """
    A minimal super-resolution network with pixel shuffle.
    """
    def __init__(self, up_factor=4):
        super(TinyUpscaler, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3 * (up_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_factor)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x
