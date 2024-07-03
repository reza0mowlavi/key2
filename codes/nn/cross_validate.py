from ._utils import Parallel

import numpy as np
import torch


def _extract_train_test_indices(X, y, cv):
    train_indices = []
    test_indices = []
    for train_idx, test_idx in cv.split(X, y):
        train_indices.append(train_idx)
        test_indices.append(test_idx)

    return train_indices, test_indices


def _extract_y_true_and_y_score(outputs):
    y_true = []
    y_score = []
    for output in outputs:
        y_true.append(output["y_true"])
        y_score.append(output["y_score"])

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    return y_true, y_score


def cross_validate(
    X, y, model, cv, metrics, n_jobs=-1, pass_train_test_data=False, inlier_label=None
):

    train_indices, test_indices = (
        _extract_train_test_indices(X=X, y=y, cv=cv)
        if inlier_label is None
        else _extract_train_test_indices_one_class(
            X=X, y=y, cv=cv, inlier_label=inlier_label
        )
    )

    if pass_train_test_data:
        outputs = Parallel(model, n_jobs=n_jobs)(
            dict(
                X_train=X[train_idx],
                y_train=y[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                train_idx=train_idx,
                test_idx=test_idx,
                counter=counter,
            )
            for counter, (train_idx, test_idx) in enumerate(
                zip(train_indices, test_indices)
            )
        )
    else:
        outputs = Parallel(model, n_jobs=n_jobs)(
            dict(train_idx=train_idx, test_idx=test_idx, counter=counter)
            for counter, (train_idx, test_idx) in enumerate(
                zip(train_indices, test_indices)
            )
        )

    if metrics is None:
        return outputs

    y_true, y_score = _extract_y_true_and_y_score(outputs)

    logs = {
        name: metric(y_true=y_true, y_score=y_score) for name, metric in metrics.items()
    }

    return logs, outputs


def _extract_train_test_indices_one_class(X, y, cv, inlier_label):
    train_indices = []
    test_indices = []
    for train_idx, test_idx in cv.split(X, y):
        train_idx = [idx for idx in train_idx if y[idx] == inlier_label]

        train_indices.append(train_idx)
        test_indices.append(test_idx)

    return train_indices, test_indices
