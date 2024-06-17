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
        self.patch_idx2path = self._create_patch_idx2path()
        self.sample_idx2path = {j: path for j, path in enumerate(self.path2data)}

    def _create_patch_idx2path(self):
        idx2path = {}
        counter = 0
        for path, count in self.path2count.items():
            for num in range(count):
                idx2path[counter] = (num, path)
                counter += 1
        return idx2path

    def _cache_data(self):
        return {path: self.read_fn(path) for path in self.path2label}

    def _extract_window(self, path, num):
        if num < self.path2count[path] - 1:
            start_index = self.hop_size * num
            end_index = start_index + self.window_size
        else:
            end_index = None
            start_index = -self.window_size
        sample = self.path2data[path][start_index:end_index]
        return sample

    def _count_samples(self):
        count_fn = lambda x: int((x - self.window_size) / self.hop_size) + 1
        return {path: count_fn(len(data)) for path, data in self.path2data.items()}

    @property
    def num_samples(self):
        return len(self.path2label)

    def get_sample(self, idx, return_path=False):
        path = self.sample_idx2path[idx]
        label = self.path2label[path]
        samples = [
            self._extract_window(path, num) for num in range(self.path2count[path])
        ]
        return (samples, label, path) if return_path else (samples, label)

    @property
    def num_patches(self):
        return len(self.patch_idx2path)

    def get_patch(self, idx, return_path=False):
        num, path = self.patch_idx2path[idx]
        sample = self._extract_window(path=path, num=num)
        label = self.path2label[path]
        return (sample, label, path) if return_path else (sample, label)
