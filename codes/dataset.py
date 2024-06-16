from pathlib import Path
import numpy as np
import pandas as pd


def _read_csv(path):
    sample = pd.read_csv(path)
    sample = sample[["key", "HL", "IL", "PL", "RL", "CL"]].values
    return sample


class Dataset:
    def __init__(self, path2label, read_fn, path2data=None):
        self.path2label = path2label
        self.read_fn = read_fn

        self.idx2path = {j: path for j, path in enumerate(self.path2label)}
        self.path2data = path2data if path2data is not None else self._cache_data()

    def _cache_data(self):
        return {path: self.read_fn(path) for path in self.path2label}

    def __len__(self):
        return len(self.path2data)

    def __getitem__(self, idx):
        path = self.idx2path[idx]
        sample = self.path2data[path]
        label = self.path2label[path]
        return sample, label


class PatchDataset:
    def __init__(
        self,
        path2label,
        window_size,
        hop_size,
        read_fn,
        path2data=None,
    ):
        self.path2label = path2label
        self.window_size = window_size
        self.hop_size = hop_size
        self.read_fn = read_fn

        self.path2data = path2data if path2data is not None else self._cache_data()
        self.path2count = self._count_samples()
        self.idx2path = self._create_idx2path()

    def _create_idx2path(self):
        idx2path = {}
        counter = 0
        for path, count in self.path2count.items():
            for num in range(count):
                idx2path[counter] = (num, path)
                counter += 1
        return idx2path

    def _cache_data(self):
        return {path: self.read_fn(path) for path in self.path2label}

    def _count_samples(self):
        count_fn = lambda x: int((x - self.window_size) / self.hop_size) + 1
        return {path: count_fn(len(data)) for path, data in self.path2data.items()}

    def __len__(self):
        return len(self.idx2path)

    def __getitem__(self, idx):
        num, path = self.idx2path[idx]

        if num < self.path2count[path] - 1:
            start_index = self.hop_size * num
            end_index = start_index + self.window_size
        else:
            end_index = None
            start_index = -self.window_size

        sample = self.path2data[path][start_index:end_index]
        label = self.path2label[path]

        return sample, label
