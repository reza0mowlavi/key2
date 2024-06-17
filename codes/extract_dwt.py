from itertools import chain
from pathlib import Path
import pickle
import argparse

import numpy as np
import pandas as pd
import keras
import pywt

from .dataset import PatchDataset


def get_cli_args():
    parser = argparse.ArgumentParser(description="Discete Wavelet Transform.")

    # Adding mandatory arguments
    parser.add_argument("data_dir", type=str, help="Data Directory")
    parser.add_argument(
        "--window-size", type=int, default=256, help="Window Size - Default: '256'"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=".",
        help="Save Directory - Default: '.'",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=None,
        help="Hop Size - Default: 'Window Size/4'",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db1",
        help="Discrete wavelet to be used - Default: 'db1'",
    )

    args = parser.parse_args()

    return (
        Path(args.data_dir),
        args.window_size,
        Path(args.save_dir),
        (args.hop_size if args.hop_size is not None else args.window_size // 4),
        args.wavelet,
    )


def read_csv(path):
    sample = pd.read_csv(path)
    sample = sample[["HL", "IL", "PL", "RL", "CL"]].values
    return sample


def get_dataset(data_dir, is_first, window_size, hop_size):
    path2label = {}

    if is_first:
        gt = pd.read_csv(data_dir.parent / "GT_DataPD_MIT-CS1PD.csv")
        data_dir = data_dir / "MIT-CS1PD"
    else:
        gt = pd.read_csv(data_dir.parent / "GT_DataPD_MIT-CS2PD.csv")
        data_dir = data_dir / "MIT-CS2PD"

    for i in range(len(gt)):
        path2label[data_dir / gt.loc[i, "file_1"]] = gt.loc[i, "gt"]
        if is_first:
            path2label[data_dir / gt.loc[i, "file_2"]] = gt.loc[i, "gt"]

    train_set = PatchDataset(
        path2label=path2label,
        window_size=window_size,
        hop_size=hop_size,
        read_fn=read_csv,
    )

    return train_set


def create_preprocess_fn(data_dir, eps=1e-6):
    with open(data_dir.parent / "statistics.pickle", "rb") as f:
        statistics = pickle.load(f)

    median = np.array(
        [statistics[key]["median"] for key in ["HL", "IL", "PL", "RL", "CL"]],
        dtype="float32",
    ).reshape(1, -1)

    IQR = np.array(
        [statistics[key]["IQR"] for key in ["HL", "IL", "PL", "RL", "CL"]],
        dtype="float32",
    ).reshape(1, -1)

    def preprocess(sample):
        sample = (sample - median) / (IQR + eps)
        sample[np.isnan(sample)] = 0  ### Impute
        return sample

    return preprocess


def extract_subclasses(dataset, preprocess_fn):
    healthy = {}
    sick = {}
    for idx in range(dataset.num_samples):
        sample, label, path = dataset.get_sample(idx, return_path=True)
        sample = [preprocess_fn(patch) for patch in sample]
        path = path.stem
        if label == 0:
            healthy[path] = sample
        else:
            sick[path] = sample
    return healthy, sick


def _extract_path2coeff(coeffs, lens, paths):
    indices_of_sections = np.cumsum(lens[:-1])
    coeffs = [np.split(coeff, indices_of_sections, axis=0) for coeff in coeffs]
    path2coeffs = {path: [] for path in paths}
    for band in coeffs:
        for path, coeff in zip(paths, band):
            path2coeffs[path].append(coeff)
    return path2coeffs


def _extract_subclass_from_path2coeff(
    path2coeffs, healthy1, healthy2, early_pd, denovo_pd
):
    dataset = {
        "healthy1": {},
        "healthy2": {},
        "early_pd": {},
        "denovo_pd": {},
    }

    for path in path2coeffs:
        if path in healthy1:
            dataset["healthy1"][path] = path2coeffs[path]
        elif path in healthy2:
            dataset["healthy2"][path] = path2coeffs[path]
        elif path in early_pd:
            dataset["early_pd"][path] = path2coeffs[path]
        elif path in denovo_pd:
            dataset["denovo_pd"][path] = path2coeffs[path]
        else:
            raise Exception

    return dataset


def pipeline(healthy1, healthy2, early_pd, denovo_pd, wavelet, window_size=None):
    X, lens, paths = [], [], []
    for path, x in chain(
        healthy1.items(),
        healthy2.items(),
        early_pd.items(),
        denovo_pd.items(),
    ):
        X.extend(x)
        paths.append(path)
        lens.append(len(x))

    lens = np.asarray(lens)
    X = keras.utils.pad_sequences(
        X, maxlen=window_size, dtype="float64", padding="post", value=0.0
    )

    coeffs = pywt.wavedec(
        X, wavelet=wavelet, axis=1
    )  ### [L, (num_seq, dim of the band, channel)]
    path2coeffs = _extract_path2coeff(coeffs, lens=lens, paths=paths)
    dataset = _extract_subclass_from_path2coeff(
        path2coeffs=path2coeffs,
        healthy1=healthy1,
        healthy2=healthy2,
        early_pd=early_pd,
        denovo_pd=denovo_pd,
    )
    return dataset


if __name__ == "__main__":
    DATA_DIR, WINDOW_SIZE, SAVE_DIR, HOP_SIZE, WAVELET = get_cli_args()

    train_set1 = get_dataset(
        data_dir=DATA_DIR, is_first=True, window_size=WINDOW_SIZE, hop_size=HOP_SIZE
    )
    train_set2 = get_dataset(
        data_dir=DATA_DIR, is_first=False, window_size=WINDOW_SIZE, hop_size=HOP_SIZE
    )

    preprocess_fn = create_preprocess_fn(data_dir=DATA_DIR)

    healthy1, early_pd = extract_subclasses(train_set1, preprocess_fn)
    healthy2, denovo_pd = extract_subclasses(train_set2, preprocess_fn)

    dataset = pipeline(
        healthy1=healthy1,
        healthy2=healthy2,
        early_pd=early_pd,
        denovo_pd=denovo_pd,
        wavelet=WAVELET,
        window_size=WINDOW_SIZE,
    )

    with open(
        SAVE_DIR / f"dataset_W={WINDOW_SIZE}_H={HOP_SIZE}_wave={WAVELET}.pickle", "wb"
    ) as f:
        pickle.dump(dataset, f)
