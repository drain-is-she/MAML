import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss
