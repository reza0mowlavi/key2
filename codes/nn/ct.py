import os

import numpy as np
import pandas as pd

import torch
import torch.multiprocessing as mp
from torch import nn
import keras
import tensorflow as tf

import optuna

import fcntl
from pathlib import Path
from functools import partial, wraps
import pickle
from collections import defaultdict
import argparse
import json
import copy

from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold

from .dataset import PatchDataset, SubjectDataset, patch_collate_fn, sample_collate_fn
from .metrics import BinaryAccuracyMetric
from .cross_validate import cross_validate
from .dataset import StratifiedBatchSampler
from .models import Transformer, TransformerConfig
from .trainer import PatchTorchTrainer
from .layers import Normalize
from .losses import BCELoss
from .utils import (
    LRSchedulerCallback,
    CosineAnnealingWarmstart,
)


def get_cli_args():
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--direction", type=str, default=None, help="Direction")
    parser.add_argument("--balanced", type=bool, default=False, help="Balanced")
    parser.add_argument("--study-name", type=str, default=None, help="Study Name")
    parser.add_argument("--data-dir", type=str, default=None, help="Data dir")
    parser.add_argument("--save-dir", type=str, default=None, help="Save dir")
    parser.add_argument("--window-size", type=int, default=None, help="Window Size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size")
    parser.add_argument("--hop-size", type=int, default=None, help="Hop Size")
    parser.add_argument("--channels", type=str, default=0, help="Channels")
    parser.add_argument("--center", type=bool, default=False, help="Center")
    parser.add_argument("--pad-mode", type=str, default="constant", help="Pad mode")

    parser.add_argument("--n-splits", type=int, default=5, help="# of splits")
    parser.add_argument("--n-repeats", type=int, default=1, help="# of repeats")
    parser.add_argument("--n-trials", type=int, default=50, help="# of trials")
    parser.add_argument("--seed", type=int, default=1997, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=1, help="# of jobs")
    parser.add_argument("--device", type=str, default=None, help="device")
    parser.add_argument(
        "--n-startup-trials", type=int, default=10, help="# of startup trials in TPE"
    )
    parser.add_argument("--hp-path", type=str, default=None, help="HP Path")

    args = parser.parse_args()
    return (
        args.study_name,
        args.direction,
        args.data_dir,
        args.save_dir,
        args.window_size,
        args.hop_size,
        args.batch_size,
        args.channels,
        args.center,
        args.pad_mode,
        args.seed,
        args.n_splits,
        args.n_repeats,
        args.n_jobs,
        args.device,
        args.n_trials,
        args.n_startup_trials,
        args.hp_path,
        args.balanced,
    )


def remove_nan_rows(arr, axis=-1):
    nan_rows = np.isnan(arr).any(axis=axis)
    return arr[~nan_rows]


def standarize(x, mean, var, eps=1e-9):
    return (x - mean) / np.sqrt(var + eps)


def read_csv(path, channels, moments=None):
    sample = pd.read_csv(path)
    sample = sample[["HL", "IL", "PL", "RL", "CL"]].values
    sample = sample[:, channels]
    sample = remove_nan_rows(sample)
    if moments is not None:
        sample = standarize(sample, **moments)
    return sample


def convert_label_to_num(labels):
    labels = np.asarray(labels)
    labels = np.where(np.logical_or(labels == "H1", labels == "H2"), 0, 1)
    if not np.shape(labels):
        labels = float(labels)
    return labels


def collate_fn_wrapper(func):
    @wraps(func)
    def wrapper(batch, **kwds):
        batch = [(x, convert_label_to_num(y)) for x, y in batch]
        return func(batch=batch, **kwds)

    return wrapper


def extract_subject2path(path2label):
    subject2path = defaultdict(list)
    subject2label = {}
    for path, label in path2label.items():
        subject = path.stem.split(".")[1].split("_")[0]
        subject2path[subject].append(path)
        subject2label[subject] = label
    return dict(subject2path), subject2label


def extract_path2label(data_dir):
    def _get_dataset(is_first, path2label):
        if is_first:
            gt = pd.read_csv(data_dir.parent / "GT_DataPD_MIT-CS1PD.csv")
            label_dict = {False: "H1", True: "EP"}
            local_data_dir = data_dir / "MIT-CS1PD"
        else:
            gt = pd.read_csv(data_dir.parent / "GT_DataPD_MIT-CS2PD.csv")
            label_dict = {False: "H2", True: "DP"}
            local_data_dir = data_dir / "MIT-CS2PD"

        for i in range(len(gt)):
            path2label[local_data_dir / gt.loc[i, "file_1"]] = label_dict[
                gt.loc[i, "gt"]
            ]
            if is_first:
                path2label[local_data_dir / gt.loc[i, "file_2"]] = label_dict[
                    gt.loc[i, "gt"]
                ]

    _1st_path2label = {}
    _2nd_path2label = {}
    _get_dataset(is_first=True, path2label=_1st_path2label)
    _get_dataset(is_first=False, path2label=_2nd_path2label)

    return _1st_path2label, _2nd_path2label


def make_train_loader(
    path2label,
    path2data,
    window_size,
    hop_size,
    center,
    pad_mode,
    batch_size,
    device,
    dtype,
    moments=None,
):
    dataset = PatchDataset(
        path2label=path2label,
        window_size=window_size,
        hop_size=hop_size,
        center=center,
        pad_mode=pad_mode,
        path2data=path2data,
        read_fn=None,
    )

    collate_fn = partial(
        patch_collate_fn, device=device, moments=moments, dtype=dtype, padding_value=0.0
    )
    collate_fn = collate_fn_wrapper(collate_fn)

    sampler = StratifiedBatchSampler(
        targets=[dataset.patch_idx2target[idx] for idx in range(len(dataset))],
        batch_size=batch_size,
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn
    )
    return loader


def make_test_loader(
    path2label,
    path2data,
    window_size,
    hop_size,
    center,
    pad_mode,
    batch_size,
    device,
    dtype,
    moments=None,
):
    dataset = SubjectDataset(
        path2label=path2label,
        window_size=window_size,
        hop_size=hop_size,
        center=center,
        pad_mode=pad_mode,
        path2data=path2data,
        read_fn=None,
    )

    collate_fn = partial(
        sample_collate_fn,
        device=device,
        moments=moments,
        dtype=dtype,
        padding_value=0.0,
    )
    collate_fn = collate_fn_wrapper(collate_fn)

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return loader


def load_data(path2label, channels):
    path2data = {path: read_csv(path, channels=channels) for path in path2label}
    return path2data


class Trainer(PatchTorchTrainer):
    def __init__(self, model, norm=None):
        super().__init__()
        self.model = model
        self.norm = norm

    def call(self, data, training):
        self.train(bool(training))
        x = data["x"]
        input_length = data.get("input_length", None)
        if self.norm is not None:
            x = self.norm(x)
        return self.model(x=x, input_length=input_length)


def create_model(device, dtype, config, norm, balanced, **kwds):
    model = Transformer(config)
    trainer = Trainer(model, norm=norm).to(device, dtype)
    trainer.in_sequence_mode = True

    if "weight_decay" not in kwds:
        optimizer = torch.optim.Adam(model.parameters(), lr=kwds.get("lr", 0.001))
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=kwds.get("lr", 0.001),
            weight_decay=kwds["weight_decay"],
        )

    scheduler = CosineAnnealingWarmstart(
        optimizer,
        maximal_lr=kwds["maximal_lr"],
        minimal_lr=kwds["minimal_lr"],
        steps=int(kwds["epochs"] * kwds["n_iter_p_epoch"] * 0.9),
        warmup_steps=int(kwds["epochs"] * kwds["n_iter_p_epoch"] * 0.1),
        initial_lr=kwds["initial_lr"],
    )
    lr_callback = LRSchedulerCallback(scheduler)

    trainer.compile(
        optimizer,
        loss=BCELoss(from_logits=True, reduction="mean", balanced=balanced),
        metrics=[
            BinaryAccuracyMetric(threshold=0.0, name="acc"),
            keras.metrics.AUC(from_logits=True, name="auc"),
        ],
    )
    trainer.metrics[1].build([(None, 1)], [(None, 1)])
    trainer.built = True
    trainer.in_sequence_mode = True

    return trainer, lr_callback


def pipeline(
    model_kwds,
    path2label,
    path2data,
    subject2path,
    subjects,
    train_idx,
    test_idx,
    batch_size,
    window_size,
    hop_size,
    center,
    pad_mode,
    device,
    dtype,
    run_name,
    save_dir,
    balanced,
    moments=None,
    counter=None,
):
    def _extract_info(indices):
        subs = subjects[indices]
        paths = []
        for sub in subs:
            paths.extend(subject2path[sub])
        path2label_ = {path: path2label[path] for path in paths}
        path2data_ = {path: path2data[path] for path in paths}
        return path2label_, path2data_

    train_path2label, train_path2data = _extract_info(indices=train_idx)
    test_path2label, test_path2data = _extract_info(indices=test_idx)

    train_loader = make_train_loader(
        path2label=train_path2label,
        path2data=train_path2data,
        window_size=window_size,
        hop_size=hop_size,
        center=center,
        pad_mode=pad_mode,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        moments=moments,
    )

    test_loader = make_test_loader(
        path2label=test_path2label,
        path2data=test_path2data,
        window_size=window_size,
        hop_size=hop_size,
        center=center,
        pad_mode=pad_mode,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        moments=moments,
    )
    ###

    norm = None
    if model_kwds["input_norm"] == "layer":
        norm = Normalize(dim=1)
    if model_kwds["input_norm"] == "standard":
        norm = keras.layers.Normalization()
        for _, x in train_loader.dataset.path2data.items():
            norm.adapt(x)
    ###
    model_kwds["n_iter_p_epoch"] = len(train_loader)
    trainer, lr_callback = create_model(
        **model_kwds, norm=norm, dtype=dtype, device=device, balanced=balanced
    )
    callbacks = [
        lr_callback,
        keras.callbacks.TensorBoard(
            log_dir=save_dir / "tb" / run_name / str(counter),
        ),
    ]
    trainer.fit(
        train_loader,
        epochs=model_kwds["epochs"],
        callbacks=callbacks,
        verbose=0,
        validation_data=test_loader,
    )

    predictions = trainer.predict(test_loader, verbose=0)

    return {"y_score": predictions["y_pred"], "y_true": predictions["y_true"]}


def build_config(trial, hp):
    config = hp["config"]
    config["hidden_size"] = 2 ** trial.suggest_int("hidden_dim", low=3, high=8, step=1)
    config["num_layers"] = trial.suggest_int("n_layers", 1, 4, step=1)
    config["num_attention_heads"] = 2 * trial.suggest_int(
        "num_attention_heads", 1, 3, step=1
    )
    config["hidden_dropout"] = trial.suggest_float(
        "hidden_dropout", low=0.0, high=0.5, step=0.05
    )
    config["attention_dropout"] = trial.suggest_float(
        "attention_dropout", low=0.0, high=0.5, step=0.05
    )
    config["dropout_feedforward"] = config["hidden_dropout"]
    config["project_dropout"] = trial.suggest_float(
        "project_dropout", low=0.0, high=0.5, step=0.05
    )
    config["dropout_clf"] = trial.suggest_float(
        "dropout_clf", low=0.0, high=0.5, step=0.05
    )
    return TransformerConfig(**config)


def log(trial, save_dir, logs):
    run_name = "trial-%d" % trial.number
    run_dir = (
        save_dir / "tb" / run_name
    )  # This is the dir used in tensorboard_callback of optuna
    with tf.summary.create_file_writer(str(run_dir)).as_default():
        for name, value in logs.items():
            tf.summary.scalar(name, value, step=trial.number)
            trial.set_user_attr(name, value)


def objective(
    trial: optuna.Trial,
    hp,
    batch_size,
    device,
    dtype,
    n_splits,
    n_repeats,
    seed,
    n_jobs,
    path2data,
    path2label,
    subject2path,
    subject2label,
    window_size,
    hop_size,
    center,
    pad_mode,
    save_dir,
    balanced,
    moments=None,
):
    hp = copy.deepcopy(hp)
    hp["config"] = build_config(trial, hp=hp)
    hp["epochs"] = trial.suggest_int("epochs", 100, 350, step=50)
    hp["maximal_lr"] = trial.suggest_float("maximal_lr", low=1e-4, high=1e-2, log=True)
    ##############################
    metrics_ = {
        "auc": metrics.roc_auc_score,
        "avg_pr": metrics.average_precision_score,
    }
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    subjects = np.asarray(list(subject2label), dtype=object)
    labels = np.asarray([subject2label[sub] for sub in subjects], dtype=object)
    outputs = cross_validate(
        X=subjects,
        y=labels,
        model=partial(
            pipeline,
            model_kwds=hp,
            path2label=path2label,
            path2data=path2data,
            subject2path=subject2path,
            subjects=subjects,
            batch_size=batch_size,
            window_size=window_size,
            hop_size=hop_size,
            center=center,
            pad_mode=pad_mode,
            device=device,
            dtype=dtype,
            save_dir=save_dir,
            balanced=balanced,
            run_name=("trial-%d" % trial.number),
            moments=moments,
        ),
        cv=cv,
        metrics=None,
        n_jobs=n_jobs,
    )
    scores = {key: [met(**preds) for preds in outputs] for key, met in metrics_.items()}
    logs = {}
    logs.update({key: np.mean(score) for key, score in scores.items()})
    logs.update({f"val_{key}": np.var(score) for key, score in scores.items()})

    log(trial=trial, save_dir=save_dir, logs=logs)

    return logs["auc"]


def load_hp(path):
    with open(path, "rt") as f:
        hp = json.load(f)
    return hp


def update_dict(save_dir, value):
    save_dir = Path(save_dir)
    with open(save_dir, "r+b" if save_dir.exists() else "w+b") as f:
        # Acquire an exclusive lock on the file
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            if os.path.getsize(save_dir) > 0:
                # Read the existing dictionary
                shared_list = pickle.load(f)
            else:
                # Create a new dictionary if the file is empty
                shared_list = []

            # Update the dictionary
            shared_list.append(value)

            # Move the file pointer to the beginning of the file
            f.seek(0)

            # Write the updated dictionary back to the file
            pickle.dump(shared_list, f)

            # Truncate the file to the new size
            f.truncate()
        finally:
            # Release the lock
            fcntl.flock(f, fcntl.LOCK_UN)


def test(
    hp,
    params,
    batch_size,
    device,
    dtype,
    path2data,
    ###
    train_path2label,
    ###
    test_path2label,
    ###
    window_size,
    hop_size,
    center,
    pad_mode,
    save_dir,
    seed,
    balanced,
    moments=None,
):
    hp = copy.deepcopy(hp)
    ### build config
    config = hp["config"]
    config["hidden_size"] = 2 ** params["hidden_dim"]
    config["num_layers"] = params["n_layers"]
    config["num_attention_heads"] = 2 * params["num_attention_heads"]
    config["hidden_dropout"] = params["hidden_dropout"]
    config["attention_dropout"] = params["attention_dropout"]
    config["dropout_feedforward"] = params["hidden_dropout"]
    config["project_dropout"] = params["project_dropout"]
    config["dropout_clf"] = params["dropout_clf"]
    hp["config"] = TransformerConfig(**config)

    hp["epochs"] = params["epochs"]
    hp["maximal_lr"] = params["maximal_lr"]
    ##############################
    ### Make dataloader
    train_path2data = {path: path2data[path] for path in train_path2label}
    train_loader = make_train_loader(
        path2label=train_path2label,
        path2data=train_path2data,
        window_size=window_size,
        hop_size=hop_size,
        center=center,
        pad_mode=pad_mode,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        moments=moments,
    )

    test_path2data = {path: path2data[path] for path in test_path2label}
    test_loader = make_test_loader(
        path2label=test_path2label,
        path2data=test_path2data,
        window_size=window_size,
        hop_size=hop_size,
        center=center,
        pad_mode=pad_mode,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        moments=moments,
    )
    ##############################
    norm = None
    if hp["input_norm"] == "layer":
        norm = Normalize(dim=1)
    if hp["input_norm"] == "standard":
        norm = keras.layers.Normalization()
        for _, x in train_loader.dataset.path2data.items():
            norm.adapt(x)
    ###
    hp["n_iter_p_epoch"] = len(train_loader)
    trainer, lr_callback = create_model(
        **hp, norm=norm, dtype=dtype, device=device, balanced=balanced
    )
    callbacks = [
        lr_callback,
        keras.callbacks.TensorBoard(
            log_dir=save_dir / "tb" / "final",
        ),
    ]
    trainer.fit(
        train_loader,
        epochs=hp["epochs"],
        callbacks=callbacks,
        verbose=0,
        validation_data=test_loader,
    )

    predictions = trainer.predict(test_loader, verbose=0)
    return metrics.roc_auc_score(
        y_true=predictions["y_true"],
        y_score=predictions["y_pred"],
    )


if __name__ == "__main__":
    METHOD = "Transformer"
    mp.set_start_method("spawn", force=True)
    (
        STUDY_NAME,
        DIRECTION,
        DATA_DIR,
        SAVE_DIR,
        WINDOW_SIZE,
        HOP_SIZE,
        BATCH_SIZE,
        CHANNELS,
        CENTER,
        PAD_MODE,
        SEED,
        N_SPLITS,
        N_REPEATS,
        N_JOBS,
        DEVICE,
        N_TRIALS,
        N_STARTUP_TRIALS,
        HP_PATH,
        BALANCED,
    ) = get_cli_args()

    if DIRECTION not in ("12", "21"):
        raise Exception(f"DIRECTION '{DIRECTION}' must be either '12' or '21'.")

    save_dir = Path(SAVE_DIR) / STUDY_NAME
    save_dir.mkdir(exist_ok=True, parents=False)
    channels = [int(x) for x in CHANNELS.split(",")]
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ### Save initial
    info = {
        "study_name": STUDY_NAME,
        "method": METHOD,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "channels": CHANNELS,
        "n_repeats": N_REPEATS,
        "n_splits": N_SPLITS,
        "seed": SEED,
        "center": CENTER,
        "pad_mode": PAD_MODE,
        "hp_path": HP_PATH,
        "balanced": BALANCED,
        "batch_size": BATCH_SIZE,
        "n_startup_trials": N_STARTUP_TRIALS,
        "direction": DIRECTION,
    }
    with open(save_dir / "info.json", "wt") as f:
        json.dump(info, f)

    data_dir = Path(DATA_DIR)
    hp = load_hp(HP_PATH)
    hp["config"]["input_dim"] = len(channels)

    _1st_path2label, _2nd_path2label = extract_path2label(data_dir)
    train_path2label, test_path2label = (
        (_1st_path2label, _2nd_path2label)
        if DIRECTION == "12"
        else (_2nd_path2label, _1st_path2label)
    )
    train_subject2path, train_subject2label = extract_subject2path(train_path2label)
    test_subject2path, test_subject2label = extract_subject2path(test_path2label)
    path2data = load_data({**train_path2label, **test_path2label}, channels=channels)

    ###
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_STARTUP_TRIALS, multivariate=True
    )
    storage_name = f"sqlite:///{save_dir}/{STUDY_NAME}.db"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_name,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    study.optimize(
        partial(
            objective,
            hp=hp,
            path2label=train_path2label,
            path2data=path2data,
            subject2path=train_subject2path,
            subject2label=train_subject2label,
            window_size=WINDOW_SIZE,
            hop_size=HOP_SIZE,
            center=CENTER,
            pad_mode=PAD_MODE,
            batch_size=BATCH_SIZE,
            device=device,
            dtype=dtype,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            seed=SEED,
            n_jobs=N_JOBS,
            save_dir=save_dir,
            balanced=BALANCED,
        ),
        n_trials=(
            N_TRIALS - len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
        ),
        callbacks=[optuna.integration.TensorBoardCallback(save_dir / "tb", "auc")],
        show_progress_bar=True,
        n_jobs=1,
    )

    test_auc = test(
        hp=hp,
        params=study.best_params,
        batch_size=BATCH_SIZE,
        device=device,
        dtype=dtype,
        path2data=path2data,
        train_path2label=train_path2label,
        test_path2label=test_path2label,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        center=CENTER,
        pad_mode=PAD_MODE,
        save_dir=save_dir,
        seed=SEED,
        balanced=BALANCED,
        moments=None,
    )

    ### Save
    info = {
        "study_name": STUDY_NAME,
        "method": METHOD,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "channels": CHANNELS,
        "n_repeats": N_REPEATS,
        "n_splits": N_SPLITS,
        "seed": SEED,
        "center": CENTER,
        "pad_mode": PAD_MODE,
        "best_value": study.best_value,
        "study": study,
        "hp_path": HP_PATH,
        "balanced": BALANCED,
        "test_auc": test_auc,
        "batch_size": BATCH_SIZE,
        "n_startup_trials": N_STARTUP_TRIALS,
        "direction": DIRECTION,
    }
    update_dict(save_dir.parent / f"logs_{DIRECTION}.pickle", info)
