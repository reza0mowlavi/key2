import numpy as np

import torch
from torch import nn

from keras import Metric


class MeanMetric(Metric):
    def __init__(self, name, dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.total = 0
        self.count = 0

    @torch.inference_mode()
    def update_state(self, y):
        self.count += y.size(0) if y.dim() > 0 else 1
        self.total += y.sum().cpu().item()

    def result(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    def reset_state(self):
        self.total = 0
        self.count = 0


class AccuracyMetric(Metric):
    def __init__(self, name="acc", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.count = 0
        self.correct = 0

    @torch.inference_mode()
    def update_state(self, y_true, y_pred):
        y_true = y_true.flatten()

        y_pred = y_pred.argmax(-1)
        y_pred = y_pred.flatten()

        self.count += y_true.size(0)
        self.correct += (y_pred == y_true).sum().cpu().item()

    def result(self):
        if self.count == 0:
            return 0
        return self.correct / self.count

    def reset_state(self):
        self.count = 0
        self.correct = 0


class BinaryAccuracyMetric(Metric):
    def __init__(self, threshold, name="acc", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.threshold = threshold
        self.count = 0
        self.correct = 0

    @torch.inference_mode()
    def update_state(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        y_pred = y_pred >= self.threshold

        self.count += y_true.size(0)
        self.correct += (y_pred == y_true).sum().cpu().item()

    def result(self):
        if self.count == 0:
            return 0
        return self.correct / self.count

    def reset_state(self):
        self.count = 0
        self.correct = 0


class BinaryClassificationScore(Metric):
    def __init__(self, threshold, beta=1, name="BinaryFBetaScore", **kwds) -> None:
        super().__init__(name=name, **kwds)
        self.threshold = threshold
        self.beta = float(beta)

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.additional_names = ["precision", "recall", "acc", "f_beta"]

    def reset_state(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        return super().reset_state()

    @torch.inference_mode()
    def update_state(self, y_true, y_pred, *args, **kwds):
        y_pred = y_pred > self.threshold
        y_pred = y_pred.cpu().numpy().flatten()
        y_true = y_true.cpu().numpy().flatten()

        self.tp += np.sum(y_pred * y_true)
        self.tn += np.sum((1 - y_pred) * (1 - y_true))
        self.fp += np.sum(y_pred * (1 - y_true))
        self.fn += np.sum((1 - y_pred) * y_true)

    def recall(self):
        numerator = self.tp
        denominator = self.tp + self.fn
        return numerator / denominator if denominator != 0 else 0

    def precision(self):
        numerator = self.tp
        denominator = self.tp + self.fp
        return numerator / denominator if denominator != 0 else 0

    def inverse_recall(self):
        numerator = self.tp + self.fn
        denominator = self.tp
        return numerator / denominator if denominator != 0 else 0

    def inverse_precision(self):
        numerator = self.tp + self.fp
        denominator = self.tp
        return numerator / denominator if denominator != 0 else 0

    def f_beta(self):
        inverse_recall = self.inverse_recall()
        inverse_precision = self.inverse_precision()
        numerator = 1 + self.beta**2
        denominator = self.beta**2 * inverse_recall + inverse_precision
        return numerator / denominator if denominator != 0 else 0

    def accuracy(self):
        numerator = self.tp + self.tn
        denominator = self.tn + self.tp + self.fn + self.fp
        return numerator / denominator if denominator != 0 else 0

    def result(self):
        precision = self.precision()
        recall = self.recall()
        accuracy = self.accuracy()
        f_beta = self.f_beta()
        return {
            "precision": precision,
            "recall": recall,
            "acc": accuracy,
            "f_beta": f_beta,
        }

    def get_config(self):
        return {
            **super().get_config(),
            "threshold": self.threshold,
            "beta": self.beta,
        }
