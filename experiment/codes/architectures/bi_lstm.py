from typing import Any

import torch
import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(
        self,
        lstm_depth: int,
        lstm_hidden: int,
        num_lead: int = 12,
        backbone_out_dim: int = 512,
    ) -> None:
        super(BiLSTM, self).__init__()

        self.lstm_hidden = lstm_hidden
        self.bilstm = nn.LSTM(
            num_lead, lstm_hidden, num_layers=lstm_depth, bidirectional=True)

        self.fc = nn.Linear(lstm_hidden*2, backbone_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size, num_lead, seqlen -> seqlen, batch_size, num_lead
        x = x.permute(2, 0, 1)
        feat, _ = self.bilstm(x)

        # feat: seqlen, batch_size, lstm_hidden*2 -> batch_size, lstm_hidden*2, seqlen
        feat = feat.permute(1, 2, 0)
        feat = nn.AdaptiveAvgPool1d(1)(feat)
        feat = feat.squeeze(-1)
        feat = self.fc(feat)
        return feat

def _bi_lstm(
    arch: str,
    lstm_depth: int,
    lstm_hidden: int,
    **kwargs: Any,
) -> BiLSTM:
    model = BiLSTM(lstm_depth, lstm_hidden, **kwargs)
    return model

def lstm_d1_h64(**kwargs: Any) -> BiLSTM:
    lstm_depth = 1
    lstm_hidden = 64
    return _bi_lstm("lstm_d1_h64", lstm_depth, lstm_hidden, **kwargs)

def lstm_d1_h128(**kwargs: Any) -> BiLSTM:
    lstm_depth = 1
    lstm_hidden = 128
    return _bi_lstm("lstm_d1_h128", lstm_depth, lstm_hidden, **kwargs)

def lstm_d1_h256(**kwargs: Any) -> BiLSTM:
    lstm_depth = 1
    lstm_hidden = 256
    return _bi_lstm("lstm_d1_h256", lstm_depth, lstm_hidden, **kwargs)

def lstm_d2_h64(**kwargs: Any) -> BiLSTM:
    lstm_depth = 2
    lstm_hidden = 64
    return _bi_lstm("lstm_d2_h64", lstm_depth, lstm_hidden, **kwargs)

def lstm_d2_h128(**kwargs: Any) -> BiLSTM:
    lstm_depth = 2
    lstm_hidden = 128
    return _bi_lstm("lstm_d2_h128", lstm_depth, lstm_hidden, **kwargs)

def lstm_d2_h256(**kwargs: Any) -> BiLSTM:
    lstm_depth = 2
    lstm_hidden = 256
    return _bi_lstm("lstm_d2_h256", lstm_depth, lstm_hidden, **kwargs)

def lstm_d3_h64(**kwargs: Any) -> BiLSTM:
    lstm_depth = 3
    lstm_hidden = 64
    return _bi_lstm("lstm_d3_h64", lstm_depth, lstm_hidden, **kwargs)

def lstm_d3_h128(**kwargs: Any) -> BiLSTM:
    lstm_depth = 3
    lstm_hidden = 128
    return _bi_lstm("lstm_d3_h128", lstm_depth, lstm_hidden, **kwargs)

def lstm_d3_h256(**kwargs: Any) -> BiLSTM:
    lstm_depth = 3
    lstm_hidden = 256
    return _bi_lstm("lstm_d3_h256", lstm_depth, lstm_hidden, **kwargs)
