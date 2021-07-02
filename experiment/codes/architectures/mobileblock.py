import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.architectures.effnet_utils import (
    MemoryEfficientSwish, Swish,
    get_same_padding_conv1d, drop_connect
)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block (MobilenetV2)
    From https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
    """
    def __init__(self, block_args, global_params):
        super(MBConvBlock, self).__init__()

        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None)\
                      and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        Conv1d = get_same_padding_conv1d(sequence_length=global_params.seqlen)

        # number of input channels
        inp = self._block_args.input_filters
        # number of output channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio

        # Expansion phase
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv1d(in_channels=inp, out_channels=oup,
                                       kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm1d(num_features=oup,
                                       momentum=self._bn_mom, eps=self._bn_eps)
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride

        self._depthwise_conv = Conv1d(
            in_channels=oup, out_channels=oup, groups=oup,
            # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm1d(num_features=oup,
                                   momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            sq = self._block_args.input_filters * self._block_args.se_ratio
            num_squeezed_channels = max(1, int(sq))
            self._se_reduce = Conv1d(in_channels=oup,
                                     out_channels=num_squeezed_channels,
                                     kernel_size=1)
            self._se_expand = Conv1d(in_channels=num_squeezed_channels,
                                     out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv1d(in_channels=oup, out_channels=final_oup,
                                    kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm1d(num_features=final_oup,
                                   momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            #print("MB excite pre", x.size())
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))
        #print("MB bn1", x.size())

        # Squeeze and Excitation
        if self.has_se:
            #print("MB se pre", x.size())
            x_squeezed = F.adaptive_avg_pool1d(x, 1)
            x_squeezed = self._se_expand(
                self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
            #print("MB se post", x.size())

        x = self._bn2(self._project_conv(x))
        #print("MB bn2", x.size())

        # Skip connection and drop connect
        input_filters = self._block_args.input_filters
        output_filters = self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 \
           and input_filters == output_filters:
            #print("MB skip pre drop", x.size())
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate,
                                 training=self.training)
            #print("MB skip connect pre", x.size())
            #print("MB skip connect, input", inputs.size())
            x = x + inputs  # skip connection
            #print("MB skip connect", x.size())
        return x

    def set_swish(self, memory_efficient=True):
        """
        Sets swish function as memory efficient (for training)
        or standard (for export)
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
