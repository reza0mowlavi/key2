import torch
from torch import nn


def compute_class_weight(y):
    counts = torch.bincount(y.to(torch.int32))
    num_samples = len(y)
    weights = (num_samples) / (counts * 2)
    return weights


class BCELoss(nn.Module):
    def __init__(self, from_logits, reduction="mean", balanced=None):
        super().__init__()
        self.from_logits = from_logits
        self.reduction = reduction
        self.balanced = balanced
        self.loss_fn = (
            nn.functional.binary_cross_entropy_with_logits
            if from_logits
            else nn.functional.binary_cross_entropy
        )

    def forward(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        sample_weight = None
        if self.balanced:
            class_weight = compute_class_weight(y_true)
            sample_weight = torch.where(y_true == 0, class_weight[0], class_weight[1])

        loss = self.loss_fn(
            input=y_pred,
            target=y_true,
            weight=sample_weight,
            reduction=self.reduction,
        )

        return loss
