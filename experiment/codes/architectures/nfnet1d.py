"""
Adapted from `https://github.com/vballoli/nfnets-pytorch/blob/main/nfnets/models/resnet.py`
"""
from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import Tensor, nn

from codes.architectures.nfnet_utils import ScaledStdConv1d

def conv1x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    base_conv: nn.Conv1d = ScaledStdConv1d
) -> nn.Conv1d:
    """1x3 convolution with padding"""
    return base_conv(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=dilation, bias=False,
                     dilation=dilation)

def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    base_conv: nn.Conv1d = ScaledStdConv1d
) -> nn.Conv1d:
    """1x1 convolution"""
    return base_conv(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)

class BasicBlock1d(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        base_conv: nn.Conv1d = ScaledStdConv1d
    ) -> None:
        super(BasicBlock1d, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv1x3(inplanes, planes, stride, base_conv=base_conv)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes, base_conv=base_conv)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck1d(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        base_conv: int = ScaledStdConv1d,
    ) -> None:
        super(Bottleneck1d, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width, base_conv=base_conv)
        self.conv2 = conv1x3(
            width, width, stride, groups, dilation, base_conv=base_conv)
        self.conv3 = conv1x1(
            width, planes * self.expansion, base_conv=base_conv)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:

        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class NFResNet1d(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock1d, Bottleneck1d]],
        layers: List[int],
        num_lead: int = 12,
        backbone_out_dim: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        base_conv: nn.Conv1d = ScaledStdConv1d
    ) -> None:
        super(NFResNet1d, self).__init__()

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = base_conv(num_lead, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], base_conv=base_conv)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0], base_conv=base_conv)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1], base_conv=base_conv)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2], base_conv=base_conv)
        self.avgpool = nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Linear(512 * block.expansion, backbone_out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1d):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1d):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock1d, Bottleneck1d]],
        planes: int, blocks: int,
        stride: int = 1,
        dilate: bool = False,
        base_conv: nn.Conv1d = ScaledStdConv1d
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,
                        stride, base_conv=base_conv),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, base_conv=base_conv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups,
                      base_width=self.base_width, dilation=self.dilation,
                      base_conv=base_conv))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _nf_resnet1d(
    arch: str,
    block: Type[Union[BasicBlock1d, Bottleneck1d]],
    layers: List[int],
    base_conv: nn.Conv1d,
    **kwargs: Any
) -> NFResNet1d:
    model = NFResNet1d(block, layers, base_conv=base_conv, **kwargs)
    return model

def nf_resnet1d18(
    base_conv: nn.Conv1d=ScaledStdConv1d,
    **kwargs: Any
) -> NFResNet1d:
    return _nf_resnet1d('nf_resnet1d18', BasicBlock1d, [2, 2, 2, 2],
                        base_conv=base_conv, **kwargs)

def nf_resnet1d34(
    base_conv: nn.Conv1d=ScaledStdConv1d,
    **kwargs: Any
) -> NFResNet1d:
    return _nf_resnet1d('nf_resnet1d34', BasicBlock1d, [3, 4, 6, 3],
                        base_conv=base_conv, **kwargs)


def nf_resnet1d50(
    base_conv: nn.Conv1d=ScaledStdConv1d,
    **kwargs: Any
) -> NFResNet1d:
    return _nf_resnet1d('nf_resnet1d50', Bottleneck1d, [3, 4, 6, 3],
                        base_conv=base_conv, **kwargs)

def nf_resnet1d101(
    base_conv: nn.Conv1d=ScaledStdConv1d,
    **kwargs: Any
) -> NFResNet1d:
    return _nf_resnet1d('nf_resnet1d101', Bottleneck1d, [3, 4, 23, 3],
                        base_conv=base_conv, **kwargs)
