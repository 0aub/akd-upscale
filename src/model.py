import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LightweightResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, 3, 1, 1)
        self.conv2 = DepthwiseSeparableConv(channels, channels, 3, 1, 1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return F.relu(out + identity, inplace=True)

class LightweightSRModel(nn.Module):
    """
    A lightweight image super-resolution model utilizing depthwise-separable
    convolutions and small residual blocks. 
    """
    def __init__(self, up_factor=4, base_ch=32, num_blocks=4):
        super().__init__()
        self.head = nn.Conv2d(3, base_ch, kernel_size=3, padding=1, bias=False)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(LightweightResidualBlock(base_ch))
        self.res_blocks = nn.Sequential(*blocks)

        # Final conv to get 3*(up_factor^2) channels, then pixelshuffle
        self.tail = nn.Conv2d(base_ch, 3*(up_factor**2), kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(up_factor)

    def forward(self, x):
        out = self.head(x)
        out = self.res_blocks(out)
        out = self.tail(out)
        out = self.pixel_shuffle(out)
        return out

class StrongPatchDiscriminator(nn.Module):
    """
    A stronger patch-based discriminator.
    The output is patch-wise real/fake classification at a lower spatial resolution.
    """
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        # Helper function to build a conv block
        def conv_block(ic, oc, stride=1):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        layers = []
        layers += conv_block(in_ch, base_ch, stride=2)      # downsample
        layers += conv_block(base_ch, base_ch, stride=1)
        layers += conv_block(base_ch, base_ch*2, stride=2)  # downsample
        layers += conv_block(base_ch*2, base_ch*2, stride=1)
        layers += conv_block(base_ch*2, base_ch*4, stride=2) # downsample
        layers += conv_block(base_ch*4, base_ch*4, stride=1)

        # final conv to get 1-channel output
        layers.append(nn.Conv2d(base_ch*4, 1, kernel_size=3, stride=1, padding=1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x -> [B, 1, H/8, W/8] (depending on how many strided convs)
        return self.net(x)
