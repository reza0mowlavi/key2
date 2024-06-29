import numpy as np
import librosa


def calculate_statistics(features, axis=-1):
    percentiles = np.percentile(
        features, q=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95], axis=axis
    ).transpose()
    mean = np.nanmean(features, axis=axis, keepdims=True)
    std = np.nanstd(features, axis=axis, keepdims=True)
    var = np.nanvar(features, axis=axis, keepdims=True)
    rms = np.nanmean(np.sqrt(features**2), axis=axis, keepdims=True)
    return np.concatenate([percentiles, mean, std, var, rms], axis=axis)


def calculate_crossings(features, mean=None, axis=-1):
    zero_crossing_indices = np.diff(features > 0, axis=axis)
    no_zero_crossings = zero_crossing_indices.sum(axis=axis, keepdims=True)

    mean = (
        mean.reshape(-1, 1)
        if mean is not None
        else np.nanmean(features, axis=axis, keepdims=True)
    )

    mean_crossing_indices = np.diff(features > mean, axis=axis)
    no_mean_crossings = mean_crossing_indices.sum(axis=axis, keepdims=True)
    return np.concatenate([no_zero_crossings, no_mean_crossings], axis=axis)


def dwt_features(list_coeff, channel, start_level=0):
    def _get_features(features):
        statistics = calculate_statistics(features)
        crossings = calculate_crossings(features, mean=statistics[:, -4])
        return np.concatenate([crossings, statistics], axis=-1)

    features_per_level = []
    for level in range(start_level, len(list_coeff)):
        coeff = list_coeff[level]
        features_per_level.append(_get_features(coeff[:, :, channel]))

    features = np.concatenate(features_per_level, axis=-1)
    return features


def stft_features(spectrals, channel, feature_type=2, log=True):
    spectrals = spectrals[:, :, channel]
    power = np.abs(spectrals) ** 2
    power = librosa.power_to_db(power) if log else power

    if feature_type == 0:
        return power

    statistics = calculate_statistics(power)
    crossings = calculate_crossings(power, mean=statistics[:, -4])

    if feature_type == 1:
        return np.concatenate([crossings, statistics], axis=-1)
    if feature_type == 2:
        return np.concatenate([power, crossings, statistics], axis=-1)

    raise Exception(f"Feature type {feature_type} is not knowns.")
