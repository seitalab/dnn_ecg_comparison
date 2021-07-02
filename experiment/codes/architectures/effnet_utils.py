import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def round_filters(filters, global_params):
    """
    Calculate and round number of filters based on depth multiplier.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """
    Round number of filters based on depth multiplier.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


# Drop connect
def drop_connect(inputs, p, training):
    """
    Drop connect.
    inputs: [batchsize, dim, seqlen]
    """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1],
                                dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv1d(sequence_length=None):
    """
    Chooses static padding if you have specified an sequence length,
    and dynamic padding otherwise.
    Static padding is necessary for ONNX exporting of models. """
    if sequence_length is None:
        return Conv1dDynamicSamePadding
    else:
        return partial(Conv1dStaticSamePadding,
                       sequence_length=sequence_length)

class Conv1dDynamicSamePadding(nn.Conv1d):
    """ 1D Convolutions like TensorFlow, for a dynamic sequence_length"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, 0, dilation, groups, bias)
        if len(self.stride) != 1:
            self.stride = [self.stride[0]]

    def forward(self, x):
        isq = x.size()[-1:]
        ksq = self.weight.size()[-1:]
        ssq = self.stride[-1]
        osq = math.ceil(isq / ssq)
        pad_sq = max((osq - 1) * self.stride[-1]\
                    + (ksq - 1) * self.dilation[0] + 1 - isq, 0)
        if pad_sq > 0:
            x = F.pad(x, [pad_sq // 2, pad_sq - pad_sq // 2])
        return F.conv1d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv1dStaticSamePadding(nn.Conv1d):
    """ 1D Convolutions like TensorFlow, for a fixed sequence_length"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 sequence_length=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        if len(self.stride) != 1:
            self.stride = [self.stride[0]]

        # Calculate padding based on sequence length and save it
        assert sequence_length is not None
        isq = sequence_length
        ksq = self.weight.size()[-1]
        ssq = self.stride[-1]
        osq = math.ceil(isq / ssq)
        pad_sq = max((osq - 1) * self.stride[-1] \
                     + (ksq - 1) * self.dilation[0] + 1 - isq, 0)

        if pad_sq > 0:
            pad_left, pad_right = pad_sq//2, pad_sq - pad_sq//2
            self.static_padding = ZeroPad1d((pad_left, pad_right))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv1d(x, self.weight, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return x

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class ZeroPad1d(nn.Module):

    def __init__(self, pad):
        super(ZeroPad1d, self).__init__()
        self.pad_l = int(pad[0])
        self.pad_r = int(pad[1])

    def forward(self, input):
        bs = input.size(0)
        pad_l = torch.zeros([bs, input.size(1), self.pad_l])
        pad_l = pad_l.to(input.device)
        if self.pad_r == 0:
            return torch.cat([pad_l, input])

        pad_r = torch.zeros([bs, input.size(1), self.pad_r])
        pad_r = pad_r.to(input.device)
        return torch.cat([pad_l, input, pad_r], dim=-1)
