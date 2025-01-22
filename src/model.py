import torch.nn as nn
import torch.nn.functional as F


# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.depthwise = nn.Conv2d(
#             in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False
#         )
#         self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x

# class LightweightResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = DepthwiseSeparableConv(channels, channels, 3, 1, 1)
#         self.conv2 = DepthwiseSeparableConv(channels, channels, 3, 1, 1)

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.conv1(x), inplace=True)
#         out = self.conv2(out)
#         return F.relu(out + identity, inplace=True)

# class LightweightSRModel(nn.Module):
#     """
#     A lightweight image super-resolution model utilizing depthwise-separable
#     convolutions and small residual blocks. 
#     """
#     def __init__(self, up_factor=4, base_ch=32, num_blocks=4):
#         super().__init__()
#         self.head = nn.Conv2d(3, base_ch, kernel_size=3, padding=1, bias=False)

#         blocks = []
#         for _ in range(num_blocks):
#             blocks.append(LightweightResidualBlock(base_ch))
#         self.res_blocks = nn.Sequential(*blocks)

#         # Final conv to get 3*(up_factor^2) channels, then pixelshuffle
#         self.tail = nn.Conv2d(base_ch, 3*(up_factor**2), kernel_size=3, padding=1, bias=False)
#         self.pixel_shuffle = nn.PixelShuffle(up_factor)

#     def forward(self, x):
#         out = self.head(x)
#         out = self.res_blocks(out)
#         out = self.tail(out)
#         out = self.pixel_shuffle(out)
#         return out


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

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, 3, 1, 1)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x

class Generator(nn.Module):
    def __init__(self, up_factor=4, base_ch=32):
        super().__init__()
        self.encoder1 = UNetBlock(3, base_ch)
        self.encoder2 = UNetBlock(base_ch, base_ch * 2)
        self.encoder3 = UNetBlock(base_ch * 2, base_ch * 4)

        self.bottleneck = UNetBlock(base_ch * 4, base_ch * 8)

        # Channel alignment layers for skip connections
        self.align3 = nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=1, stride=1, bias=False)
        self.align2 = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=1, stride=1, bias=False)
        self.align1 = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1, stride=1, bias=False)

        self.decoder3 = UNetBlock(base_ch * 4, base_ch * 4)
        self.decoder2 = UNetBlock(base_ch * 2, base_ch * 2)
        self.decoder1 = UNetBlock(base_ch, base_ch)

        self.head = nn.Conv2d(base_ch, 3 * (up_factor**2), kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(up_factor)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)                           # [B, base_ch, H, W]
        enc2 = self.encoder2(self.pool(enc1))             # [B, base_ch*2, H/2, W/2]
        enc3 = self.encoder3(self.pool(enc2))             # [B, base_ch*4, H/4, W/4]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))     # [B, base_ch*8, H/8, W/8]

        # Decoder
        up_bottleneck = F.interpolate(bottleneck, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        up_bottleneck = self.align3(up_bottleneck)        # Align channels to match enc3
        dec3 = self.decoder3(up_bottleneck + enc3)        # Skip connection with enc3

        up_dec3 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        up_dec3 = self.align2(up_dec3)                    # Align channels to match enc2
        dec2 = self.decoder2(up_dec3 + enc2)              # Skip connection with enc2

        up_dec2 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        up_dec2 = self.align1(up_dec2)                    # Align channels to match enc1
        dec1 = self.decoder1(up_dec2 + enc1)              # Skip connection with enc1

        # Output
        out = self.head(dec1)
        out = self.pixel_shuffle(out)
        return out

def conv_block(ic, oc, stride=1):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, stride, 1),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        layers = []
        layers += conv_block(in_ch, base_ch, stride=2) # downsample
        layers += conv_block(base_ch, base_ch, stride=1)
        layers += conv_block(base_ch, base_ch*2, stride=2) # downsample
        layers += conv_block(base_ch*2, base_ch*2, stride=1)
        layers += conv_block(base_ch*2, base_ch*4, stride=2) # downsample
        layers += conv_block(base_ch*4, base_ch*4, stride=1)

        # final conv to get 1-channel output
        layers.append(nn.Conv2d(base_ch*4, 1, kernel_size=3, stride=1, padding=1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x -> [B, 1, H/8, W/8] (depending on how many strided convs)
        return self.net(x)



if __name__ == "__main__":
    model = Generator(up_factor=4, base_ch=32)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("base_ch=32:", num_params)

    model = Generator(up_factor=4, base_ch=40)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("base_ch=40:", num_params)

    model = Generator(up_factor=4, base_ch=48)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("base_ch=48:", num_params)

    model = Generator(up_factor=4, base_ch=64)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("base_ch=64:", num_params)
