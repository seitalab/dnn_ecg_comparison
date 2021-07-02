from typing import Optional, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

class ScaledStdConv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        gain: bool = True,
        gamma: float = 1.0,
        eps: float = 1e-5,
        use_layernorm: bool = False
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(
            self.out_channels, 1, 1)) if gain else None
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps

        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * \
                F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            mean = torch.mean(
                self.weight, dim=[1, 2], keepdim=True)
            std = torch.std(
                self.weight, dim=[1, 2], keepdim=True, unbiased=True)
            weight = self.scale * (self.weight - mean) / (std + self.eps)

        if self.gain is not None:
            weight = weight * self.gain

        return weight

    def forward(self, x):
        return F.conv1d(x, self.get_weight(),  self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
