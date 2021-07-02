import re
import collections
from typing import Any, List

import torch
import torch.nn as nn

from codes.architectures.mobileblock import MBConvBlock
from codes.architectures.effnet_utils import (
    round_filters, round_repeats, get_same_padding_conv1d,
    drop_connect, Swish, MemoryEfficientSwish
)

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'drop_connect_rate', "seqlen"])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

class BlockDecoder(object):
    """
    Block Decoder for readability,
    straight from the official TensorFlow repository
    """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and\
                 options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d' % block.strides,
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks
        inside the network.
        :param string_list: a list of strings,
                            each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
            #print(blocks_args[-1])
            #input()
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet_params(width_coefficient=None, depth_coefficient=None,
                        dropout_rate=0.2, drop_connect_rate=0.2, seqlen=None):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s1_e1_i32_o16_se0.25', 'r2_k3_s2_e6_i16_o24_se0.25',
        'r2_k5_s2_e6_i24_o40_se0.25', 'r3_k3_s2_e6_i40_o80_se0.25',
        'r3_k5_s1_e6_i80_o112_se0.25', 'r4_k5_s2_e6_i112_o192_se0.25',
        'r1_k3_s1_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        seqlen=seqlen,
    )

    return blocks_args, global_params

class EfficientNet1d(nn.Module):

    def __init__(
        self,
        width: float,
        depth: float,
        res: int,
        dropout_rate: float,
        sequence_length: int,
        num_lead: int = 12,
        backbone_out_dim: int = 512,
    ) -> None:
        super(EfficientNet1d, self).__init__()

        blocks_args, global_params = efficientnet_params(
            width, depth, dropout_rate=dropout_rate,
            drop_connect_rate=dropout_rate, seqlen=sequence_length)

        assert isinstance(blocks_args, list)
        assert len(blocks_args) > 0
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv1d = get_same_padding_conv1d(sequence_length=global_params.seqlen)

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = num_lead
        # number of output channels
        out_channels = round_filters(32, self._global_params)

        self._conv_stem = Conv1d(in_channels, out_channels, kernel_size=3,
                                 stride=2, bias=False)
        self._bn0 = nn.BatchNorm1d(num_features=out_channels,
                                   momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self._global_params),
                output_filters=round_filters(block_args.output_filters,
                                             self._global_params),
                num_repeat=round_repeats(block_args.num_repeat,
                                         self._global_params)
            )

            # The first block needs to take care of stride and
            # filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args,
                                                self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv1d(in_channels, out_channels,
                                 kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm1d(num_features=out_channels,
                                   momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool1d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)

        self.fc = nn.Linear(out_channels, backbone_out_dim)

        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """
        Sets swish function as memory efficient (for training)
        or standard (for export)
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """
        Calls extract_features to extract features,
        applies final linear layer.
        """
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self._dropout(x)
        x = self.fc(x)
        return x

def _effnet1d(
    arch: str,
    effnet_params: List,
    **kwargs
) -> EfficientNet1d:
    width, depth, res, dropout_rate = effnet_params
    model = EfficientNet1d(
        width=width, depth=depth, res=res, dropout_rate=dropout_rate, **kwargs)
    return model

def effnet1d_b0(**kwargs: Any) -> EfficientNet1d:
    return _effnet1d('efficientnetb0', [1.0, 1.0, 224, 0.2], **kwargs)

def effnet1d_b1(**kwargs: Any) -> EfficientNet1d:
    return _effnet1d('efficientnetb1', [1.0, 1.1, 240, 0.2], **kwargs)

def effnet1d_b2(**kwargs: Any) -> EfficientNet1d:
    return _effnet1d('efficientnetb2', [1.1, 1.2, 260, 0.3], **kwargs)

def effnet1d_b3(**kwargs: Any) -> EfficientNet1d:
    return _effnet1d('efficientnetb3', [1.2, 1.4, 300, 0.3], **kwargs)

def effnet1d_b4(**kwargs: Any) -> EfficientNet1d:
    return _effnet1d('efficientnetb4', [1.4, 1.8, 380, 0.4], **kwargs)

def effnet1d_b5(**kwargs: Any) -> EfficientNet1d:
    return _effnet1d('efficientnetb5', [1.6, 2.2, 456, 0.4], **kwargs)

def effnet1d_b6(**kwargs: Any) -> EfficientNet1d:
    return _effnet1d('efficientnetb6', [1.8, 2.6, 528, 0.5], **kwargs)

def effnet1d_b7(**kwargs: Any) -> EfficientNet1d:
    return _effnet1d('efficientnetb7', [2.0, 3.1, 600, 0.5], **kwargs)

def effnet1d_b8(**kwargs: Any) -> EfficientNet1d:
    return _effnet1d('efficientnetb8', [2.2, 3.6, 672, 0.5], **kwargs)
