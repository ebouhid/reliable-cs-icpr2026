"""Class-balancing / data-augmentation methods."""

from __future__ import annotations

import random

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE

from reliable_cs.config import GNI_NOISE_STD, RANDOM_SEED


def random_undersampling(X, y, random_state: int = RANDOM_SEED):
    X = np.array(X)
    y = np.array(y)
    np.random.seed(random_state)

    n_forest = np.sum(y == "Forest")
    n_nonforest = np.sum(y == "Non-forest")

    if n_forest > n_nonforest:
        maj_class, min_class = "Forest", "Non-forest"
    else:
        maj_class, min_class = "Non-forest", "Forest"

    n_to_keep = np.sum(y == min_class)
    idx_majority = np.where(y == maj_class)[0]
    idx_minority = np.where(y == min_class)[0]
    idx_majority_selected = np.random.choice(idx_majority, n_to_keep, replace=False)
    idx_final = np.concatenate([idx_majority_selected, idx_minority])
    np.random.shuffle(idx_final)
    return X[idx_final], y[idx_final]


def ascending_entropy_undersampling(forest_features, non_forest_features, y_forest, y_non_forest):
    min_len = min(len(forest_features), len(non_forest_features))
    X_bal = []
    y_bal = []
    for idx in range(min_len):
        X_bal.append(forest_features[idx])
        X_bal.append(non_forest_features[idx])
        y_bal.append(y_forest[idx])
        y_bal.append(y_non_forest[idx])
    return np.float64(X_bal), y_bal


def descending_entropy_undersampling(forest_features, non_forest_features, y_forest, y_non_forest):
    min_len = min(len(forest_features), len(non_forest_features))
    X_bal = []
    y_bal = []
    for idx in range(min_len - 1, -1, -1):
        X_bal.append(forest_features[idx])
        X_bal.append(non_forest_features[idx])
        y_bal.append(y_forest[idx])
        y_bal.append(y_non_forest[idx])
    return np.float64(X_bal), y_bal


def gaussian_noise_injection_da(
    X, y, noise_std: float = GNI_NOISE_STD, random_state: int = RANDOM_SEED
):
    np.random.seed(random_state)
    random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    n_forest = np.sum(y == "Forest")
    n_nonforest = np.sum(y == "Non-forest")

    if n_forest >= n_nonforest:
        min_class = "Non-forest"
        diff = n_forest - n_nonforest
    else:
        min_class = "Forest"
        diff = n_nonforest - n_forest

    X_min = X[y == min_class]
    idx_selected = np.random.choice(len(X_min), diff, replace=True)
    X_selected = X_min[idx_selected]
    noise = np.random.normal(0, noise_std, X_selected.shape)
    X_synth = X_selected + noise
    X_bal = np.vstack([X, X_synth])
    y_bal = np.concatenate([y, np.array([min_class] * diff)])
    return X_bal, y_bal


def smote(X, y, random_state: int = RANDOM_SEED):
    sampler = SMOTE(random_state=random_state)
    return sampler.fit_resample(X, y)


def adasyn(X, y, random_state: int = RANDOM_SEED):
    sampler = ADASYN(random_state=random_state)
    return sampler.fit_resample(X, y)


def build_balanced_sets(data, seed: int = RANDOM_SEED) -> dict[str, tuple[list, list]]:
    """Apply all balancing methods on median-entropy-filtered data (paper default flow)."""
    n = len(data.campaign.train_files)
    xs: dict[str, list] = {
        "random_undersample": [],
        "ascending_entropy_undersample": [],
        "descending_entropy_undersample": [],
        "gni": [],
        "smote": [],
        "adasyn": [],
    }
    ys: dict[str, list] = {k: [] for k in xs}

    for i in range(n):
        x_ru, y_ru = random_undersampling(
            data.x_filtered_scaled[i], data.y_filtered[i], random_state=seed
        )
        xs["random_undersample"].append(x_ru)
        ys["random_undersample"].append(y_ru)

        x_asc, y_asc = ascending_entropy_undersampling(
            data.x_forest_filtered_scaled[i],
            data.x_nonforest_filtered_scaled[i],
            data.y_forest_filtered[i],
            data.y_nonforest_filtered[i],
        )
        xs["ascending_entropy_undersample"].append(x_asc)
        ys["ascending_entropy_undersample"].append(y_asc)

        x_desc, y_desc = descending_entropy_undersampling(
            data.x_forest_filtered_scaled[i],
            data.x_nonforest_filtered_scaled[i],
            data.y_forest_filtered[i],
            data.y_nonforest_filtered[i],
        )
        xs["descending_entropy_undersample"].append(x_desc)
        ys["descending_entropy_undersample"].append(y_desc)

        x_gni, y_gni = gaussian_noise_injection_da(
            data.x_filtered_scaled[i], data.y_filtered[i], random_state=seed
        )
        xs["gni"].append(x_gni)
        ys["gni"].append(y_gni)

        x_sm, y_sm = smote(data.x_filtered_scaled[i], data.y_filtered[i], random_state=seed)
        xs["smote"].append(x_sm)
        ys["smote"].append(y_sm)

        x_ad, y_ad = adasyn(data.x_filtered_scaled[i], data.y_filtered[i], random_state=seed)
        xs["adasyn"].append(x_ad)
        ys["adasyn"].append(y_ad)

    return {k: (xs[k], ys[k]) for k in xs}
