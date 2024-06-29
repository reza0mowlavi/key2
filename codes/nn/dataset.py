from collections import defaultdict

import numpy as np
import torch
from torch import nn


def _randomized_rounding(x):
    eps = np.random.rand() - 0.5
    return np.round(x + eps).astype("int32")


class StratifiedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, targets, batch_size, drop_last=False, weights=None):
        super().__init__()
        self.targets = targets
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.indices = self._make_indices()
        self.weights = self._make_weights() if weights is None else weights

    def _make_weights(self):
        label2len = {label: len(val) for label, val in self.indices.items()}
        total_len = sum(label2len.values())

        weights = {label: label2len[label] / total_len for label in label2len}
        return weights

    def _make_indices(self):
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, target in enumerate(self.targets):
            class_indices[target].append(idx)

        class_indices = {key: np.array(value) for key, value in class_indices.items()}

        return class_indices

    def _num_of_samples_per_batch(self, batch_size):
        s = 0
        num_samples = {}
        for label in self.weights:
            num_samples[label] = _randomized_rounding(self.weights[label] * batch_size)
            s += num_samples[label]

        if s > batch_size:
            num_samples[label] -= s - batch_size
        elif s < batch_size:
            num_samples[label] += batch_size - s

        return num_samples

    def __iter__(self):
        # Shuffle the indices and return iterator
        rnd_indices = {
            label: np.random.permutation(len(self.indices[label])).astype("int32")
            for label in self.weights
        }
        indices = {
            label: self.indices[label][rnd] for label, rnd in rnd_indices.items()
        }

        start_indices = {label: 0 for label in self.weights}

        for _ in range(len(self)):
            num_samples = self._num_of_samples_per_batch(self.batch_size)
            batch_indices = []

            for label in self.weights:
                start_idx = start_indices[label]
                end_idx = start_idx + num_samples[label]
                batch_indices.extend(indices[label][start_idx:end_idx])
                start_indices[label] = end_idx

            yield batch_indices

    def __len__(self):
        n = len(self.targets) / self.batch_size
        n = int(n) if self.drop_last else int(np.ceil(n))
        return n


def normalize(inputs, mean, var):
    return (inputs - mean) / var
