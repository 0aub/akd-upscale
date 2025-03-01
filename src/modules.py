import torch
from torch import nn
import torch.nn.functional as F
import math

# ref: https://github.com/changzy00/pytorch-attention

"""
PyTorch implementation of Squeeze-and-Excitation Networks

As described in https://arxiv.org/pdf/1709.01507

The SE block is composed of two main components: the squeeze layer and the excitation layer.
The squeeze layer reduces the spatial dimensions of the input feature maps by taking the average
value of each channel. This reduces the number of parameters in the network, making it more efficient.
The excitation layer then applies a learnable gating mechanism to the squeezed feature maps, which helps
to select the most informative channels and amplifies their contribution to the final output.

"""

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

"""
PyTorch implementation of CBAM: Convolutional Block Attention Module

As described in https://arxiv.org/pdf/1807.06521

The attention mechanism is achieved by using two different types of attention gates:
channel-wise attention and spatial attention. The channel-wise attention gate is applied
to each channel of the input feature map, and it allows the network to focus on the most
important channels based on their spatial relationships. The spatial attention gate is applied
to the entire input feature map, and it allows the network to focus on the most important regions
of the image based on their channel relationships.
"""

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

"""
PyTorch implementation of Bam: Bottleneck attention module

As described in http://bmvc2018.org/contents/papers/0092.pdf

Given a 3D feature map, BAM produces a 3D attention map to emphasize important elements. BAM
decomposes the process of inferring a 3D attention map in two streams , so that the
computational and parametric overhead are significantly reduced. As the channels of feature
maps can be regarded as feature detectors, the two branches (spatial and channel) explicitly
learn "what" and "where" to focus on.
"""

class ChannelGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.mlp(y)
        y = self.bn(y).view(b, c, 1, 1)
        return y.expand_as(x)

class SpatialGate(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3, dilation_val=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(channel // reduction, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.bn(y)
        return y.expand_as(x)

class BAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_attn = ChannelGate(channel)
        self.spatial_attn = SpatialGate(channel)

    def forward(self, x):
        attn = F.sigmoid(self.channel_attn(x) + self.spatial_attn(x))
        return x + x * attn

"""
PyTorch implementation of A2-Nets: Double Attention Networks

As described in https://arxiv.org/pdf/1810.11579

The component is designed with a double attention mechanism in two steps, where the first step
gathers features from the entire space into a compact set through second-order
attention pooling and the second step adaptively selects and distributes features
to each location via another attention.
"""

class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n):

        super().__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.proj = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, x):
        b, c, h, w = x.shape
        A = self.convA(x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)
        tmpA = A.view(b, self.c_m, h * w)
        attention_maps = B.view(b, self.c_n, h * w)
        attention_vectors = V.view(b, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        tmpZ = tmpZ.view(b, self.c_m, h, w)
        tmpZ = self.proj(tmpZ)
        return tmpZ

"""
PyTorch implementation of Srm : A style-based recalibration module for
convolutional neural networks

As described in https://arxiv.org/pdf/1903.10829

SRM first extracts the style information from each channel of the feature maps by style pooling,
then estimates per-channel recalibration weight via channel-independent style integration.
By incorporating the relative importance of individual styles into feature maps,
SRM effectively enhances the representational ability of a CNN.
"""

class SRM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, groups=channel,
                             bias=False)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, h, w = x.shape
        # style pooling
        mean = x.reshape(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.reshape(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat([mean, std], dim=-1)
        # style integration
        z = self.cfc(u)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)

"""
PyTorch implementation of Gcnet: Non-local networks meet squeeze-excitation networks and beyond

As described in https://arxiv.org/pdf/1904.11492

GC module contains three steps: (a) a context modeling module which aggregates the features of all positions
together to form a global context feature; (b) a feature transform module to capture the channel-wise
interdependencies; and (c) a fusion module to merge the global context feature into features of all positions.
"""

class GCModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.LayerNorm([channel // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1)
        )

    def context_modeling(self, x):
        b, c, h, w = x.shape
        input_x = x
        input_x = input_x.reshape(b, c, h * w)
        context = self.conv(x)
        context = context.reshape(b, 1, h * w).transpose(1, 2)
        out = torch.matmul(input_x, context)
        out = out.reshape(b, c, 1, 1)
        return out

    def forward(self, x):
        context = self.context_modeling(x)
        y = self.transform(context)
        return x + y

"""
PyTorch implementation of Selective Kernel Networks

As described in https://arxiv.org/abs/1903.06586

A building block called Selective Kernel (SK) unit is designed, in which multiple
branches with different kernel sizes are fused using softmax attention that is guided
by the information in these branches. Different attentions on these branches yield
different sizes of the effective receptive fields of neurons in the fusion layer.
"""

class SKLayer(nn.Module):
    def __init__(self, inplanes, planes, max_groups=32, ratio=16):
        super().__init__()
        groups = self.find_largest_divisor(inplanes, max_groups)

        d = max(planes // ratio, groups)
        self.planes = planes
        self.split_3x3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.split_5x5 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=2, dilation=2, groups=groups),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, d),
            nn.BatchNorm1d(d),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(d, planes)
        self.fc2 = nn.Linear(d, planes)

    @staticmethod
    def find_largest_divisor(n, max_groups):
        """
        Finds the largest divisor of n that is less than or equal to max_groups.
        """
        for i in range(min(n, max_groups), 0, -1):
            if n % i == 0:
                return i
        return 1

    def forward(self, x):
        batch_size = x.shape[0]
        u1 = self.split_3x3(x)
        u2 = self.split_5x5(x)
        u = u1 + u2
        s = self.avgpool(u).flatten(1)
        z = self.fc(s)
        attn_scores = torch.cat([self.fc1(z), self.fc2(z)], dim=1)
        attn_scores = attn_scores.view(batch_size, 2, self.planes)
        attn_scores = attn_scores.softmax(dim=1)
        a = attn_scores[:,0].view(batch_size, self.planes, 1, 1)
        b = attn_scores[:,1].view(batch_size, self.planes, 1, 1)
        u1 = u1 * a.expand_as(u1)
        u2 = u2 * b.expand_as(u2)
        x = u1 + u2
        return x

"""
PyTorch implementation of Linear Context Transform Block

As described in https://arxiv.org/pdf/1909.03834v2

Linear Context Transform (LCT) block divides all channels into different groups
and normalize the globally aggregated context features within each channel group,
reducing the disturbance from irrelevant channels. Through linear transform of
the normalized context features, LCT models global context for each channel independently.
"""

class LCT(nn.Module):
    def __init__(self, channels, groups, eps=1e-5):
        super().__init__()
        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.w = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size = x.shape[0]
        y = self.avgpool(x).view(batch_size, self.groups, -1)
        mean = y.mean(dim=-1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=-1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_norm = y_norm.reshape(batch_size, self.channels, 1, 1)
        y_norm = self.w.reshape(1, -1, 1, 1) * y_norm + self.b.reshape(1, -1, 1, 1)
        y_norm = self.sigmoid(y_norm)
        return x * y_norm.expand_as(x)

"""
PyTorch implementation of Gated Channel Transformation for Visual Recognition (GCT)

As described in http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.pdf

GCT introduces a channel normalization layer to reduce the number of parameters and
computational complexity. This lightweight layer incorporates a simple ` 2 normalization,
enabling our transformation unit applicable to operator-level without much increase of additional parameters.
"""

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate

"""
PyTorch implementation of ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks

As described in https://arxiv.org/abs/1910.03151

ECANet proposes a local crosschannel interaction strategy without dimensionality reduction,
which can be efficiently implemented via 1D convolution.
"""

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

"""
PyTorch implementation of Rotate to Attend: Convolutional Triplet Attention Module

As described in http://arxiv.org/pdf/2010.03045

Triplet attention, a novel method for computing attention weights by capturing crossdimension
interaction using a three-branch structure. For an input tensor, triplet attention builds inter-dimensional
dependencies by the rotation operation followed by residual transformations and encodes inter-channel and spatial
information with negligible computational overhead.
"""

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ks):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=1,
                              padding=(ks - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x_max = x.max(dim=1, keepdim=True)[0]
        return torch.cat([x_mean, x_max], dim=1)

class AttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = ZPool()
        self.conv = BasicConv2d(2, 1, kernel_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        y = self.compress(x)
        y = self.conv(y)
        y = self.activation(y)
        return x * y

class TripletAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.ch = AttentionGate(kernel_size)
        self.cw = AttentionGate(kernel_size)
        self.hw = AttentionGate(kernel_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ch = self.ch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # c and h
        x_cw = self.cw(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_hw = self.hw(x)
        return 1 / 3 * (x_ch + x_cw + x_hw)

"""
PyTorch implementation of Gaussian Context Transformer

As described in http://openaccess.thecvf.com//content/CVPR2021/papers/Ruan_Gaussian_Context_Transformer_CVPR_2021_paper.pdf

Gaussian Context Transformer (GCT), which achieves contextual feature excitation using
a Gaussian function that satisfies the presupposed relationship.
"""

class GCT(nn.Module):
    def __init__(self, channels, c=2, eps=1e-5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps
        self.c = c

    def forward(self, x):
        y = self.avgpool(x)
        mean = y.mean(dim=1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_transform = torch.exp(-(y_norm ** 2 / 2 * self.c))
        return x * y_transform.expand_as(x)

"""
PyTorch implementation of Coordinate Attention for Efficient Mobile Network Design

As described in https://arxiv.org/abs/2103.02907

the coordinate attention factorizes channel attention into two 1D feature encoding processes
that aggregate features along the two spatial directions, respectively. In this way, long-range
dependencies can be captured along one spatial direction and meanwhile precise positional information
can be preserved along the other spatial direction.
"""

class CoordinateAttention(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden_dim = max(8, in_dim // reduction)
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        b,c,h,w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose(-1, -2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(-1, -2)
        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)
        out = identity * a_h * a_w
        return out

"""
PyTorch implementation of SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks

As described in http://proceedings.mlr.press/v139/yang21o/yang21o.pdf

SimAM, inspired by neuroscience theories in the mammalian brain.
"""

class simam(nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super().__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

"""
PyTorch implementation of Dual Attention Network for Scene Segmentation

As described in https://arxiv.org/pdf/1809.02983.pdf

Dual Attention Network (DANet) to adaptively integrate local features with their
global dependencies based on the self-attention mechanism.
"""

class PAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.b = nn.Conv2d(dim, dim, 1)
        self.c = nn.Conv2d(dim, dim, 1)
        self.d = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, c, h, w = x.shape
        B = self.b(x).flatten(2).transpose(1, 2)
        C = self.c(x).flatten(2)
        D = self.d(x).flatten(2).transpose(1, 2)
        attn = (B @ C).softmax(dim=-1)
        y = (attn @ D).transpose(1, 2).reshape(n, c, h, w)
        out = self.alpha * y + x
        return out

class CAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_ = x.flatten(2)
        attn = torch.matmul(x_, x_.transpose(1, 2))
        attn = attn.softmax(dim=-1)
        x_ = (attn @ x_).reshape(b, c, h, w)
        out = self.beta * x_ + x
        return out