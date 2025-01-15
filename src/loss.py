import torch
import torch.nn as nn
from torchvision import models

# https://github.com/AquibPy/Super-Resolution-GAN/blob/main/loss.py
# https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/autoencoding/lpips/loss/lpips.py

class VGGLoss(nn.Module):
    """
    A perceptual loss using VGG19 up to layer 36 (conv5_4).
    Expects inputs in range [0,1] or a normalized range (typical for super-res).
    """
    def __init__(self, device):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:36].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()

    def forward(self, input_img, target_img):
        input_feats  = self.vgg(input_img)
        target_feats = self.vgg(target_img)
        return self.criterion(input_feats, target_feats)

class ScalingLayer(nn.Module):
    """
    Shifts input by a constant and then scales it (LPIPS standard).
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.tensor([[-0.030, -0.088, -0.188]]).view(1,3,1,1))
        self.register_buffer("scale", torch.tensor([[0.458, 0.448, 0.450]]).view(1,3,1,1))

    def forward(self, x):
        return (x - self.shift.to(x.device)) / self.scale.to(x.device)


class NetLinLayer(nn.Module):
    """A 1x1 conv that linearly reweights channels for LPIPS distance."""
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = []
        if use_dropout:
            layers.append(nn.Dropout())
        layers.append(nn.Conv2d(chn_in, chn_out, kernel_size=1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VGG16FeatureExtractor(nn.Module):
    """
    Extracts intermediate features from VGG16 in 5 slices:
       relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
    """
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[vgg16[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[vgg16[i] for i in range(4,9)])
        self.slice3 = nn.Sequential(*[vgg16[i] for i in range(9,16)])
        self.slice4 = nn.Sequential(*[vgg16[i] for i in range(16,23)])
        self.slice5 = nn.Sequential(*[vgg16[i] for i in range(23,30)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]


def normalize_tensor(x, eps=1e-10):
    """Channel-wise normalization of a feature map."""
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)
    return x / norm_factor


def spatial_average(x, keepdim=True):
    """Averages over spatial dimensions (H,W)."""
    return x.mean([2,3], keepdim=keepdim)


class LPIPS(nn.Module):
    """
    The LPIPS distance (Learned Perceptual Image Patch Similarity).
    Uses a fixed VGG16 backbone, then linearly combines L2 distances in feature space.
    """
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.vgg = VGG16FeatureExtractor()
        # VGG16 channels: [64,128,256,512,512]
        self.lins = nn.ModuleList([
            NetLinLayer(64, 1, use_dropout=use_dropout),
            NetLinLayer(128,1, use_dropout=use_dropout),
            NetLinLayer(256,1, use_dropout=use_dropout),
            NetLinLayer(512,1, use_dropout=use_dropout),
            NetLinLayer(512,1, use_dropout=use_dropout),
        ])
        # load official LPIPS weights here

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """
        pred, target: [B,3,H,W], typically in [0,1].
        Returns a 1D tensor of LPIPS distances per sample in batch.
        """
        pred_scaled   = self.scaling_layer(pred)
        target_scaled = self.scaling_layer(target)

        pred_feats   = self.vgg(pred_scaled)
        target_feats = self.vgg(target_scaled)

        distances = []
        for i in range(len(pred_feats)):
            feat_p = normalize_tensor(pred_feats[i])
            feat_t = normalize_tensor(target_feats[i])
            diff   = (feat_p - feat_t)**2
            diff   = self.lins[i](diff)
            diff   = spatial_average(diff, keepdim=True)
            distances.append(diff)

        total_distance = distances[0]
        for i in range(1, len(distances)):
            total_distance += distances[i]

        return total_distance.view(-1)

class Loss(nn.Module):
    """
    A single interface for these loss_name options:
      - "l1": L1 Loss
      - "mse": PyTorch's built-in MSELoss
      - "vgg": Perceptual loss using VGG19 up to layer 36
      - "lpips": Learned perceptual similarity
      - "bce": Binary cross-entropy with logits
      - "mix": A fixed weighted sum of L1 + VGG + LPIPS + BCE

    Usage in training loop (example):
      loss_fn = Loss(loss_name="combined")  # or "lpips", "l1", etc.
      out = model(lr_img)
      loss = loss_fn(out, teacher_img)
      loss.backward()
      optimizer.step()
    """
    def __init__(self, cfg):
        super().__init__()
        self.loss_name = cfg.loss.lower()

        if self.loss_name == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_name == "vgg":
            self.criterion = VGGLoss(cfg.device)
        elif self.loss_name == "lpips":
            self.criterion = LPIPS()
        elif self.loss_name == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_name == "mix":
            # We'll define fixed weights for sub-losses here:
            self.weight_l1 = cfg.weight_l1
            self.weight_vgg = cfg.weight_vgg
            self.weight_lpips = cfg.weight_lpips
            self.weight_bce = cfg.weight_bce

            self.l1_loss = nn.L1Loss()
            self.vgg_loss = VGGLoss()
            self.lpips_loss = LPIPS()
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss_name: {self.loss_name}")

    def forward(self, pred, target):
        """
        - For "combined", we sum:
             total = 1.0 * L1 + 0.1 * VGG + 0.05 * LPIPS + 0.05 * BCE
          The BCE also uses (pred, target) on pixel space. 
          (Unusual for SR, but you've requested it.)

        - Otherwise, we do the single specified criterion.
        """
        if self.loss_name in ["l1", "l2", "mse", "vgg", "lpips", "bce"]:
            return self.criterion(pred, target)

        elif self.loss_name == "mix":
            # Weighted sum of sub-losses
            val_l1    = self.l1_loss(pred, target)
            val_vgg   = self.vgg_loss(pred, target)
            val_lpips = torch.mean(self.lpips_loss(pred, target))
            val_bce   = self.bce_loss(pred, target)

            total = (self.weight_l1    * val_l1
                   + self.weight_vgg   * val_vgg
                   + self.weight_lpips * val_lpips
                   + self.weight_bce   * val_bce)
            return total

        else:
            raise NotImplementedError(f"Unknown loss_name: {self.loss_name}")
