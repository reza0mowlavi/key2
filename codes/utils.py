from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RBF(BaseEstimator, TransformerMixin):
    def __init__(self, gamma, dtype="float64", transform_on_distance=None):
        super().__init__()
        self.gamma = gamma
        self._dtype = dtype
        self.transform_on_distance = transform_on_distance

    def fit(self, X, y=None):
        if self.gamma == "scale":
            self._gamma = 1 / (X.shape[-1] * X.var())
        elif self.gamma == "auto":
            self._gamma = 1 / (X.shape[-1])
        else:
            self._gamma = self.gamma

        return self

    def transform(self, X):
        og_X = X
        X = X if self.transform_on_distance is None else self.transform_on_distance(X)
        X = X.astype(self._dtype)
        if X is og_X:
            X = np.copy(X)
        np.multiply(X, -self._gamma, X)
        np.exp(X, X)
        return X


class FeatureDataset:
    def __init__(self, features, targets, idx2path):
        self.features = features
        self.idx2path = idx2path

        self.subjects, self.subject2indices = self._exatrct_subjects()

        targets = np.asarray(targets)
        self.targets = {
            subject: targets[self.subject2indices[subject][0]]
            for subject in self.subjects
        }
        self.subject_labels = np.asarray(
            [self.targets[subject] for subject in self.subjects]
        )

    def _exatrct_subjects(self):
        subjects = set()
        subject2indices = defaultdict(list)

        for idx, path in self.idx2path.items():
            subject = path.stem.split(".")[1].split("_")[0]
            subjects.add(subject)
            subject2indices[subject].append(idx)

        subjects = np.sort(list(subjects))
        subject2indices = dict(subject2indices)

        return subjects, subject2indices

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        indices = self.subject2indices[subject]

        features = self.features[indices]
        if not isinstance(features, np.ndarray):
            features = np.concatenate(self.features[indices])

        target = self.targets[subject]
        targets = np.repeat(target, len(features))

        return {
            "features": features,
            "targets": targets,
            "length": len(features),
            "y_true": target,
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
