from typing import Type, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

clamp_val = 20

class LambdaConv1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nhead: int,
        dim_k: int=16,
        dim_u: int=4,
        dim_m: int=7,
    ) -> None:
        super(LambdaConv1d, self).__init__()

        self.nhead = nhead
        self.dim_k, self.dim_u, self.dim_m = dim_k, dim_u, dim_m
        self.dim_v = out_channels // nhead
        # self.temperature = temperature

        self.local_context = True if dim_m > 0 else False
        self.padding = (dim_m - 1) // 2

        self.to_queries = nn.Sequential(
            nn.Conv1d(in_channels, dim_k * nhead, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim_k * nhead)
        )
        self.to_keys = nn.Sequential(
            nn.Conv1d(in_channels, dim_k * dim_u, kernel_size=1, bias=False),
        )
        self.to_values = nn.Sequential(
            nn.Conv1d(in_channels, self.dim_v * dim_u, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.dim_v * dim_u)
        )

        if self.local_context:
            self.embedding = nn.Parameter(
                torch.randn([self.dim_k, self.dim_u, 1, dim_m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(
                torch.randn([self.dim_k, self.dim_u]), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_lead, sequence_length).
        Returns:
            out (torch.Tensor): Tensor of size (batch_size, dim, dim).
        """
        bs, n_dim, seqlen = x.size()

        queries = self.to_queries(x)
        queries = queries.view(bs, self.nhead, self.dim_k, seqlen)

        keys = self.to_keys(x)
        keys = keys.view(bs, self.dim_k, self.dim_u, seqlen)
        keys = F.softmax(keys, dim=-1)

        values = self.to_values(x)
        values = values.view(bs, self.dim_v, self.dim_u, seqlen)

        lambda_c = torch.einsum('bkum,bvum->bkv', keys, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)
        assert(np.all(np.isfinite(y_c.detach().cpu().numpy())))

        if self.local_context:
            values = values.view(bs, self.dim_u, -1, seqlen)
            lambda_p = F.conv2d(values, self.embedding, padding=(0, self.padding))
            lambda_p = lambda_p.view(bs, self.dim_k, self.dim_v, seqlen)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.contiguous().view(bs, -1, seqlen)
        return out

class LambdaBottleneck1d(nn.Module):
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int=1):
        super(LambdaBottleneck1d, self).__init__()

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.ModuleList([LambdaConv1d(planes, planes, nhead=4)])
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv2.append(nn.AvgPool1d(kernel_size=(3,), stride=stride, padding=(1,)))
        self.conv2.append(nn.BatchNorm1d(planes))
        self.conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_lead, seqlen)
        Returns:

        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class LambdaResNet1d(nn.Module):

    def __init__(
        self,
        block: Type[LambdaBottleneck1d],
        num_blocks: List[int],
        num_lead: int = 12,
        backbone_out_dim: int = 512,
    ) -> None:
        super(LambdaResNet1d, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(
            num_lead, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_planes)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * block.expansion, backbone_out_dim)
        )

    def _make_layer(
        self,
        block: Type[LambdaBottleneck1d],
        planes: int,
        num_blocks: int,
        stride: int=1
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_lead, sequence_length).
        Returns:
            out (torch.Tensor): Tensor of size (batch_size, backbone_out_dim).
        # """
        x = torch.clamp(x, min=-1*clamp_val, max=clamp_val)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = torch.clamp(out, min=-1*clamp_val, max=clamp_val)
        out = self.layer2(out)
        out = torch.clamp(out, min=-1*clamp_val, max=clamp_val)
        out = self.layer3(out)
        out = torch.clamp(out, min=-1*clamp_val, max=clamp_val)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def _lambdaresnet1d(
    arch: str,
    block: Type[LambdaBottleneck1d],
    layers: List[int],
    **kwargs: Any
) -> LambdaResNet1d:
    model = LambdaResNet1d(block, layers, **kwargs)
    return model

def lambda_resnet1d18(**kwargs: Any) -> LambdaResNet1d:
    """LambdaResNet-18 model from
    `"LAMBDANETWORKS: MODELING LONG-RANGE INTERACTIONS WITHOUT ATTENTION" <https://openreview.net/pdf?id=xTJEN-ggl1b>`.
    Coverted for 1d data.
    Args:

    """
    return _lambdaresnet1d(
        'LambdaResNet18', LambdaBottleneck1d, [2, 2, 2, 2], **kwargs)

def lambda_resnet1d50(**kwargs: Any) -> LambdaResNet1d:
    """LambdaResNet-50 model from
    `"LAMBDANETWORKS: MODELING LONG-RANGE INTERACTIONS WITHOUT ATTENTION" <https://openreview.net/pdf?id=xTJEN-ggl1b>`.
    Coverted for 1d data.
    Args:

    """
    return _lambdaresnet1d(
        'LambdaResNet50', LambdaBottleneck1d, [3, 4, 6, 3], **kwargs)

def lambda_resnet1d101(**kwargs: Any) -> LambdaResNet1d:
    """LambdaResNet-101 model from
    `"LAMBDANETWORKS: MODELING LONG-RANGE INTERACTIONS WITHOUT ATTENTION" <https://openreview.net/pdf?id=xTJEN-ggl1b>`.
    Coverted for 1d data.
    Args:

    """
    return _lambdaresnet1d(
        'LambdaResNet101', LambdaBottleneck1d, [3, 4, 23, 3], **kwargs)

def lambda_resnet1d152(**kwargs: Any) -> LambdaResNet1d:
    """LambdaResNet-152 model from
    `"LAMBDANETWORKS: MODELING LONG-RANGE INTERACTIONS WITHOUT ATTENTION" <https://openreview.net/pdf?id=xTJEN-ggl1b>`.
    Coverted for 1d data.
    Args:

    """
    return _lambdaresnet1d(
        'LambdaResNet152', LambdaBottleneck1d, [3, 8, 36, 3], **kwargs)
