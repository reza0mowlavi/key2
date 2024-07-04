from pathlib import Path
from collections import defaultdict
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
        center=False,
        pad_mode="constant",
        path2data=None,
    ):
        self.path2label = path2label
        self.window_size = window_size
        self.hop_size = hop_size
        self.read_fn = read_fn
        self.center = center
        self.pad_mode = pad_mode

        self.path2data = path2data if path2data is not None else self._cache_data()
        self.path2count = self._count_samples()
        self.patch_idx2path = self._create_patch_idx2path()
        self.sample_idx2path = {j: path for j, path in enumerate(self.path2data)}
        self.sample_idx2target = {
            idx: self.path2label[path] for idx, path in self.sample_idx2path.items()
        }
        self.patch_idx2target = {
            idx: self.path2label[path]
            for idx, (_, path) in self.sample_idx2path.items()
        }

    def _create_patch_idx2path(self):
        idx2path = {}
        counter = 0
        for path, count in self.path2count.items():
            for num in range(count):
                idx2path[counter] = (num, path)
                counter += 1
        return idx2path

    def _cache_data(self):
        path2data = {path: self.read_fn(path) for path in self.path2label}
        if self.center:
            path2data = {
                path: self._centerize(data) for path, data in path2data.items()
            }
        return path2data

    def _centerize(self, data):
        pad_size = self.window_size // 2
        pad_size = ((self.window_size, self.window_size), (0, 0))
        return np.pad(data, pad_size, mode=self.pad_mode)

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

    def get_patch(self, idx, return_path_num=False):
        num, path = self.patch_idx2path[idx]
        sample = self._extract_window(path=path, num=num)
        label = self.path2label[path]
        return (sample, label, num, path) if return_path_num else (sample, label)


class FeatureDataset:
    def __init__(self, list_of_features, labels):
        self.list_of_features = list_of_features
        self.labels = labels

    def __len__(self):
        return len(self.list_of_features)

    def __getitem__(self, idx):
        features = self.list_of_features[idx]
        length = len(features)
        y_true = self.labels[idx]
        targets = np.repeat(y_true, len(features))

        return {
            "features": features,
            "targets": targets,
            "length": length,
            "y_true": y_true,
        }

    def get(self, indices=None):
        indices = range(len(self)) if indices is None else indices
        batch = [self[idx] for idx in indices]

        features = np.concatenate([x["features"] for x in batch], axis=0)
        targets = np.concatenate([x["targets"] for x in batch], axis=0)
        lengths = np.asarray([x["length"] for x in batch])
        y_true = np.asarray([x["y_true"] for x in batch])

        return {
            "features": features,
            "targets": targets,
            "lengths": lengths,
            "y_true": y_true,
        }


class GramDataset:
    def __init__(self, gram, names, labels):
        self.gram = gram
        self.names = names
        self.labels = labels

        (
            self.subjects,
            self.subject_labels,
            self.subject2indices,
            self.subject2length,
        ) = self._exatrct_subjects()

    def _exatrct_subjects(self):
        subjects = set()
        subject2indices = defaultdict(list)

        for idx, name in enumerate(self.names):
            subject = name.split(".")[1].split("_")[0]
            subjects.add(subject)
            subject2indices[subject].append(idx)

        subjects = np.sort(list(subjects))
        subject_labels = np.asarray(
            [self.labels[subject2indices[subject][0]] for subject in subjects]
        )
        subject2indices = dict(subject2indices)

        subject2length = {
            subject: len(subject2indices[subject]) for subject in subjects
        }

        return subjects, subject_labels, subject2indices, subject2length

    def __len__(self):
        return len(self.subjects)

    def _extract_indices(self, indices):
        subjects = self.subjects[indices]
        subject_labels = self.subject_labels[indices]
        indices = np.concatenate(
            [self.subject2indices[subject] for subject in subjects]
        )
        return indices, subjects, subject_labels

    def get(self, indices, feature_indices):
        indices, subjects, subject_labels = self._extract_indices(indices)

        feature_indices, _, _ = self._extract_indices(feature_indices)

        features = self.gram[indices][:, feature_indices]

        y_true = subject_labels
        lengths = np.fromiter(
            (self.subject2length[subject] for subject in subjects), dtype="int32"
        )
        targets = np.repeat(y_true, lengths)

        return {
            "features": features,
            "targets": targets,
            "lengths": lengths,
            "y_true": y_true,
        }
