import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import time
from .modules import *


def get_attention(attention_name, channels=512, reduction=16, kernel_size=7):
    if attention_name == 'se_layer':
        attention_module = SELayer(channels, reduction)
    elif attention_name == 'cbam':
        attention_module = CBAM(channels, reduction, kernel_size)
    elif attention_name == 'bam':
        attention_module = BAM(channels)
    elif attention_name == 'double_attention':
        attention_module = DoubleAttention(channels, channels // reduction, channels // reduction)
    elif attention_name == 'srm':
        attention_module = SRM(channels)
    elif attention_name == 'gc_module':
        attention_module = GCModule(channels, reduction)
    elif attention_name == 'sk_layer':
        attention_module = SKLayer(channels, channels)
    elif attention_name == 'lct':
        attention_module = LCT(channels, channels // reduction)
    elif attention_name == 'gct':
        attention_module = GCT(channels)
    elif attention_name == 'eca':
        attention_module = ECA(channels, reduction)
    elif attention_name == 'triplet_attention':
        attention_module = TripletAttention(kernel_size)
    elif attention_name == 'coordinate_attention':
        attention_module = CoordinateAttention(channels, channels, reduction)
    elif attention_name == 'simam':
        attention_module = simam(channels)
    elif attention_name == 'pam':
        attention_module = PAM(channels)
    elif attention_name == 'cam':
        attention_module = CAM()
    else:
        raise ValueError(f"[ERROR]  Unsupported attention type: {attention_name}")
    return attention_module

# -----------------------------------------------------------------------------
# Basic Building Blocks
# -----------------------------------------------------------------------------
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

# Original UNet Block (used when generator_block_type == "unet")
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, 3, 1, 1)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x

# -----------------------------------------------------------------------------
# Residual Dense Block (RDB)
# -----------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16, num_layers=4):
        """
        A simple RDB:
          - Each layer outputs `growth_rate` feature maps.
          - Features are concatenated and then fused with a 1x1 convolution.
        """
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.convs = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            self.convs.append(
                nn.Conv2d(current_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True)
            )
            current_channels += growth_rate
        self.lff = nn.Conv2d(current_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        inputs = x
        concat_features = x
        for conv in self.convs:
            out = F.relu(conv(concat_features), inplace=True)
            concat_features = torch.cat([concat_features, out], dim=1)
        out = self.lff(concat_features)
        return out + inputs

# -----------------------------------------------------------------------------
# Generator Block Helper
# -----------------------------------------------------------------------------
def generator_block(in_ch, out_ch, block_type, rdb_num_layers=4, rdb_growth_rate=16):
    """
    Returns a block that adjusts channels from in_ch to out_ch.
    - If block_type=="unet", returns a UNetBlock.
    - If block_type=="rdb", first applies a conv (to adjust channels)
      and then an RDB.
    """
    if block_type == "unet":
        return UNetBlock(in_ch, out_ch)
    elif block_type == "rdb":
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            ResidualDenseBlock(out_ch, growth_rate=rdb_growth_rate, num_layers=rdb_num_layers)
        )
    else:
        raise ValueError("Unsupported generator block type")

# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, up_factor=4, base_ch=32, attention_type=None, generator_block_type="unet",
                 use_advanced_upsampling=False, rdb_num_layers=4,  rdb_growth_rate=16):
        super().__init__()
        # Encoder
        self.encoder1 = generator_block(3, base_ch, generator_block_type, rdb_num_layers, rdb_growth_rate)
        self.encoder2 = generator_block(base_ch, base_ch * 2, generator_block_type, rdb_num_layers, rdb_growth_rate)
        self.encoder3 = generator_block(base_ch * 2, base_ch * 4, generator_block_type, rdb_num_layers, rdb_growth_rate)
        # Bottleneck
        self.bottleneck = generator_block(base_ch * 4, base_ch * 8, generator_block_type, rdb_num_layers, rdb_growth_rate)
        # Optional attention after bottleneck
        if attention_type is not None:
            self.gen_attention = get_attention(attention_type, channels=base_ch * 8)
        else:
            self.gen_attention = None
        # Channel alignment layers for skip connections
        self.align3 = nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=1, stride=1, bias=False)
        self.align2 = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=1, stride=1, bias=False)
        self.align1 = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1, stride=1, bias=False)
        # Decoder
        self.decoder3 = generator_block(base_ch * 4, base_ch * 4, generator_block_type, rdb_num_layers, rdb_growth_rate)
        self.decoder2 = generator_block(base_ch * 2, base_ch * 2, generator_block_type, rdb_num_layers, rdb_growth_rate)
        self.decoder1 = generator_block(base_ch, base_ch, generator_block_type, rdb_num_layers, rdb_growth_rate)
        # Optional advanced upsampling refinement block
        if use_advanced_upsampling:
            self.upsample_refinement = generator_block(base_ch, base_ch, generator_block_type, rdb_num_layers, rdb_growth_rate)
        else:
            self.upsample_refinement = None
        # Final head and PixelShuffle upscaling
        self.head = nn.Conv2d(base_ch, 3 * (up_factor ** 2), kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(up_factor)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)                          # [B, base_ch, H, W]
        enc2 = self.encoder2(self.pool(enc1))            # [B, base_ch*2, H/2, W/2]
        enc3 = self.encoder3(self.pool(enc2))            # [B, base_ch*4, H/4, W/4]
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))    # [B, base_ch*8, H/8, W/8]
        if self.gen_attention is not None:
            bottleneck = self.gen_attention(bottleneck)
        # Decoder path with skip connections and channel alignment
        up_bottleneck = F.interpolate(bottleneck, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        up_bottleneck = self.align3(up_bottleneck)         # Align channels to match enc3
        dec3 = self.decoder3(up_bottleneck + enc3)         # Skip connection with enc3

        up_dec3 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        up_dec3 = self.align2(up_dec3)                     # Align channels to match enc2
        dec2 = self.decoder2(up_dec3 + enc2)               # Skip connection with enc2

        up_dec2 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        up_dec2 = self.align1(up_dec2)                     # Align channels to match enc1
        dec1 = self.decoder1(up_dec2 + enc1)               # Skip connection with enc1

        # Optional refinement before final upsampling
        if self.upsample_refinement is not None:
            dec1 = self.upsample_refinement(dec1)
        # Output head and pixel shuffle for upscaling
        out = self.head(dec1)
        out = self.pixel_shuffle(out)
        return out

# -----------------------------------------------------------------------------
# Discriminator
# -----------------------------------------------------------------------------
def conv_block(ic, oc, stride=1, use_spectral_norm=True):
    conv = nn.Conv2d(ic, oc, 3, stride, 1)
    if use_spectral_norm:
        conv = spectral_norm(conv)
    return nn.Sequential(
        conv,
        nn.LeakyReLU(0.2, inplace=True)
    )

class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, attention_type=None):
        super().__init__()
        layers = []
        layers += [conv_block(in_ch, base_ch, stride=2, use_spectral_norm=True)]
        layers += [conv_block(base_ch, base_ch, stride=1, use_spectral_norm=True)]
        layers += [conv_block(base_ch, base_ch * 2, stride=2, use_spectral_norm=True)]
        layers += [conv_block(base_ch * 2, base_ch * 2, stride=1, use_spectral_norm=True)]
        layers += [conv_block(base_ch * 2, base_ch * 4, stride=2, use_spectral_norm=True)]
        layers += [conv_block(base_ch * 4, base_ch * 4, stride=1, use_spectral_norm=True)]
        self.features = nn.Sequential(*layers)

        # Optional discriminator attention applied after feature extraction
        if attention_type is not None:
            self.disc_attention = get_attention(attention_type, channels=base_ch * 4)
        else:
            self.disc_attention = None

        # Final convolution to get 1-channel output
        self.final_conv = nn.Conv2d(base_ch * 4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.features(x)
        if self.disc_attention is not None:
            x = self.disc_attention(x)
        x = self.final_conv(x)
        return x

# -----------------------------------------------------------------------------
# Main block for parameter counting/testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Use a sample input: batch size 1, 3 channels, 64x64 image.
    sample_input = torch.randn(1, 3, 64, 64)

    configs = []
    base_ch = 64
    up_factor = 4

    # Define attention options: include None (for "no attention") and 15 options.
    attention_options = [None, "se_layer", "cbam", "bam", "double_attention", "srm",
                         "gc_module", "sk_layer", "lct", "gct", "eca",
                         "triplet_attention", "coordinate_attention", "simam", "pam", "cam"]

    # For each block type in {"unet", "rdb"}
    for block_type in ["unet", "rdb"]:
        for att in attention_options:
            for adv in [False, True]:
                if block_type == "unet":
                    # For UNet blocks, ignore RDB settings.
                    configs.append({
                        "block_type": block_type,
                        "base_ch": base_ch,
                        "attention": att if att is not None else "",
                        "advanced": adv,
                        "rdb_num_layers": "",
                        "rdb_growth": ""
                    })
                else:
                    # For RDB blocks, try two different settings: "small" and "large".
                    for (num_layers, growth_rate) in [(3, 8), (4, 16)]:
                        configs.append({
                            "block_type": block_type,
                            "base_ch": base_ch,
                            "attention": att if att is not None else "",
                            "advanced": adv,
                            "rdb_num_layers": num_layers,
                            "rdb_growth": growth_rate
                        })

    # Store results (each configuration with its measured forward-pass time and param count).
    results = []
    for cfg in configs:
        if cfg["block_type"] == "unet":
            model = Generator(
                up_factor=up_factor,
                base_ch=cfg["base_ch"],
                attention_type=cfg["attention"] if cfg["attention"] != "" else None,
                generator_block_type=cfg["block_type"],
                use_advanced_upsampling=cfg["advanced"],
                rdb_num_layers=0,
                rdb_growth_rate=0
            )
        else:
            model = Generator(
                up_factor=up_factor,
                base_ch=cfg["base_ch"],
                attention_type=cfg["attention"] if cfg["attention"] != "" else None,
                generator_block_type=cfg["block_type"],
                use_advanced_upsampling=cfg["advanced"],
                rdb_num_layers=cfg["rdb_num_layers"],
                rdb_growth_rate=cfg["rdb_growth"]
            )
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            _ = model(sample_input)
        elapsed = time.time() - start_time

        results.append({
            "block_type": cfg["block_type"],
            "base_ch": cfg["base_ch"],
            "attention": cfg["attention"],
            "advanced": cfg["advanced"],
            "rdb_num_layers": cfg["rdb_num_layers"],
            "rdb_growth": cfg["rdb_growth"],
            "params": num_params,
            "time": elapsed
        })

    # Sort the results by elapsed time (lowest to highest).
    results.sort(key=lambda x: x["time"])

    # Print table header.
    header = "{:>3} | {:>6} | {:>4} | {:>20} | {:>8} | {:>10} | {:>10} | {:>15} | {:>8}".format(
        "No", "Block", "Base", "Attention", "AdvUp", "RDB Layers", "RDB Growth", "Params", "Time(s)"
    )
    print("\n=== Sorted Generator Configurations (by Time) ===\n")
    print(header)
    print("-" * len(header))

    for idx, res in enumerate(results, start=1):
        # For UNet blocks, leave RDB settings empty.
        rdb_layers = str(res["rdb_num_layers"]) if res["block_type"] == "rdb" else ""
        rdb_growth = str(res["rdb_growth"]) if res["block_type"] == "rdb" else ""
        att_str = str(res["attention"]) if res["attention"] != "" else ""
        print("{:>3} | {:>6} | {:>4} | {:>20} | {:>8} | {:>10} | {:>10} | {:>15,} | {:>8.4f}".format(
            idx,
            res["block_type"],
            res["base_ch"],
            att_str,
            str(res["advanced"]),
            rdb_layers,
            rdb_growth,
            res["params"],
            res["time"]
        ))

    print("\n=== Finished Testing ===\n")