import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from config import EMBEDDING_DIM


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet50(weights=weights)

        # Remove the original classification layer
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # ResNet50 outputs 2048 features
        self.embedding = nn.Linear(2048, EMBEDDING_DIM)

    def forward(self, x):
        features = self.feature_extractor(x)

        # (B, 2048, 1, 1) -> (B, 2048)
        features = torch.flatten(features, 1)

        embedding = self.embedding(features)

        return embedding

    def freeze_backbone(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True 
