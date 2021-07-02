import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadModule(nn.Module):

    def __init__(self, in_dim: int, num_classes: int):
        super(HeadModule, self).__init__()

        self.fc1 = nn.Linear(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (num_batch, in_dim).
        Returns:
            feat (torch.Tensor): Tensor of size (num_batch, num_classes).
        """
        feat = F.relu(self.fc1(x))
        feat = self.bn1(feat)
        feat = self.drop1(feat)
        feat = self.fc2(feat)
        return feat

class Classifier(nn.Module):

    def __init__(self, backbone: nn.Module, prediction_heads: nn.Module):
        super(Classifier, self).__init__()

        self.backbone = backbone
        self.prediction_heads = prediction_heads

    def forward(self, x: torch.Tensor):

        feat = self.backbone(x)
        predictions = self.prediction_heads(feat)

        return predictions
