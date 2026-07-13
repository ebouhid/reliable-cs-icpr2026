"""Filtering, entropy sorting, and feature scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from reliable_cs.config import CampaignConfig
from reliable_cs.data import (
    compact_replicate_values,
    join_features_band_per_segment,
    load_campaign_tables,
    standardize_y,
)


@dataclass
class PreparedData:
    """All matrices required by the SVM experiment for one campaign."""

    campaign: CampaignConfig
    # Raw joined features (with-outliers index 0 used for PRODES)
    x_train_joined: list[pd.DataFrame]
    x_test_joined: pd.DataFrame
    # Labels
    y_prodes: list
    y_majority_raw: list[list]  # per outlier approach, includes Draw/Undefined
    y_test: list
    # Campaign (no Draw/Undefined), entropy-sorted
    x_campaign: list[np.ndarray]
    y_campaign: list[list]
    x_forest_campaign: list[list]
    y_forest_campaign: list[list]
    x_nonforest_campaign: list[list]
    y_nonforest_campaign: list[list]
    # Median-entropy filtered, entropy-sorted
    x_filtered: list[np.ndarray]
    y_filtered: list[list]
    x_forest_filtered: list[list]
    y_forest_filtered: list[list]
    x_nonforest_filtered: list[list]
    y_nonforest_filtered: list[list]
    # Scaled
    x_prodes_scaled: np.ndarray
    x_test_prodes_scaled: np.ndarray
    x_campaign_scaled: list[np.ndarray]
    x_test_campaign_scaled: list[np.ndarray]
    x_forest_campaign_scaled: list[np.ndarray]
    x_nonforest_campaign_scaled: list[np.ndarray]
    x_filtered_scaled: list[np.ndarray]
    x_test_filtered_scaled: list[np.ndarray]
    x_forest_filtered_scaled: list[np.ndarray]
    x_nonforest_filtered_scaled: list[np.ndarray]
    # Metadata for exports
    df_test: pd.DataFrame
    test_segment_ids: list


def _filter_and_sort(
    x_joined: pd.DataFrame,
    y_majority: list,
    entropies: list,
    median_threshold: float,
) -> tuple[
    np.ndarray,
    list,
    list,
    list,
    list,
    list,
    np.ndarray,
    list,
    list,
    list,
    list,
    list,
]:
    x_no_draw: list = []
    y_no_draw: list = []
    ent_no_draw: list = []
    x_filt: list = []
    y_filt: list = []
    ent_filt: list = []

    for sample_id, sample in enumerate(y_majority):
        if sample in ("Forest", "Non-forest"):
            x_no_draw.append(x_joined.iloc[sample_id])
            y_no_draw.append(sample)
            ent_no_draw.append(entropies[sample_id])
            if entropies[sample_id] <= median_threshold:
                x_filt.append(x_joined.iloc[sample_id])
                y_filt.append(sample)
                ent_filt.append(entropies[sample_id])

    x_no_draw_arr = np.float64(x_no_draw)
    x_filt_arr = np.float64(x_filt)

    df_no_draw = pd.DataFrame(
        x_no_draw_arr,
        columns=[f"Feature_{i}" for i in range(1, len(x_no_draw_arr[0]) + 1)],
    )
    df_filt = pd.DataFrame(
        x_filt_arr,
        columns=[f"Feature_{i}" for i in range(1, len(x_filt_arr[0]) + 1)],
    )
    df_no_draw.insert(0, "Entropy", ent_no_draw)
    df_filt.insert(0, "Entropy", ent_filt)
    df_no_draw["Class"] = y_no_draw
    df_filt["Class"] = y_filt

    df_no_draw = df_no_draw.sort_values(by="Entropy", ascending=True).reset_index(drop=True)
    df_filt = df_filt.sort_values(by="Entropy", ascending=True).reset_index(drop=True)

    forest = df_no_draw[df_no_draw["Class"] == "Forest"]
    nonforest = df_no_draw[df_no_draw["Class"] == "Non-forest"]
    forest_f = df_filt[df_filt["Class"] == "Forest"]
    nonforest_f = df_filt[df_filt["Class"] == "Non-forest"]

    return (
        np.float64(df_no_draw.iloc[:, 1:-1].values.tolist()),
        df_no_draw["Class"].tolist(),
        forest.iloc[:, 1:-1].values.tolist(),
        forest["Class"].tolist(),
        nonforest.iloc[:, 1:-1].values.tolist(),
        nonforest["Class"].tolist(),
        np.float64(df_filt.iloc[:, 1:-1].values.tolist()),
        df_filt["Class"].tolist(),
        forest_f.iloc[:, 1:-1].values.tolist(),
        forest_f["Class"].tolist(),
        nonforest_f.iloc[:, 1:-1].values.tolist(),
        nonforest_f["Class"].tolist(),
    )


def prepare_data(campaign: CampaignConfig) -> PreparedData:
    """Load CSVs and build all scaled train/test matrices for one campaign."""
    df_train, x_train_raw, df_test = load_campaign_tables(campaign)

    x_train_joined = [
        join_features_band_per_segment(x_train_raw[i], campaign.bands)
        for i in range(len(campaign.train_files))
    ]
    x_test_joined = join_features_band_per_segment(df_test, campaign.bands)

    entropies = [compact_replicate_values(df, "Entropy") for df in df_train]
    median_thresholds = [float(np.median(e)) for e in entropies]

    y_majority_raw: list[list] = []
    y_prodes = None
    for df in df_train:
        y_prodes_temp, y_maj = standardize_y(df, data_type="train")
        y_majority_raw.append(y_maj)
        y_prodes = y_prodes_temp  # notebook keeps labels from the last train file
    y_test, _ = standardize_y(df_test, data_type="test")

    x_campaign = []
    y_campaign = []
    x_forest_campaign = []
    y_forest_campaign = []
    x_nonforest_campaign = []
    y_nonforest_campaign = []
    x_filtered = []
    y_filtered = []
    x_forest_filtered = []
    y_forest_filtered = []
    x_nonforest_filtered = []
    y_nonforest_filtered = []

    for i in range(len(campaign.train_files)):
        (
            x_c,
            y_c,
            x_f,
            y_f,
            x_nf,
            y_nf,
            x_e,
            y_e,
            x_fe,
            y_fe,
            x_nfe,
            y_nfe,
        ) = _filter_and_sort(
            x_train_joined[i],
            y_majority_raw[i],
            entropies[i],
            median_thresholds[i],
        )
        x_campaign.append(x_c)
        y_campaign.append(y_c)
        x_forest_campaign.append(x_f)
        y_forest_campaign.append(y_f)
        x_nonforest_campaign.append(x_nf)
        y_nonforest_campaign.append(y_nf)
        x_filtered.append(x_e)
        y_filtered.append(y_e)
        x_forest_filtered.append(x_fe)
        y_forest_filtered.append(y_fe)
        x_nonforest_filtered.append(x_nfe)
        y_nonforest_filtered.append(y_nfe)

    # PRODES scaling (features from with-outliers; labels from last file, as in notebook)
    prodes_scaler = StandardScaler()
    x_prodes_scaled = prodes_scaler.fit_transform(x_train_joined[0])
    x_test_prodes_scaled = prodes_scaler.transform(x_test_joined)

    x_campaign_scaled = []
    x_test_campaign_scaled = []
    x_forest_campaign_scaled = []
    x_nonforest_campaign_scaled = []
    x_filtered_scaled = []
    x_test_filtered_scaled = []
    x_forest_filtered_scaled = []
    x_nonforest_filtered_scaled = []

    for i in range(len(campaign.train_files)):
        campaign_scaler = StandardScaler()
        campaign_scaler.fit(x_campaign[i])
        x_campaign_scaled.append(campaign_scaler.transform(x_campaign[i]))
        x_forest_campaign_scaled.append(campaign_scaler.transform(x_forest_campaign[i]))
        x_nonforest_campaign_scaled.append(campaign_scaler.transform(x_nonforest_campaign[i]))
        x_test_campaign_scaled.append(campaign_scaler.transform(x_test_joined))

        filtered_scaler = StandardScaler()
        filtered_scaler.fit(x_filtered[i])
        x_filtered_scaled.append(filtered_scaler.transform(x_filtered[i]))
        x_forest_filtered_scaled.append(filtered_scaler.transform(x_forest_filtered[i]))
        x_nonforest_filtered_scaled.append(filtered_scaler.transform(x_nonforest_filtered[i]))
        x_test_filtered_scaled.append(filtered_scaler.transform(x_test_joined))

    return PreparedData(
        campaign=campaign,
        x_train_joined=x_train_joined,
        x_test_joined=x_test_joined,
        y_prodes=y_prodes,
        y_majority_raw=y_majority_raw,
        y_test=y_test,
        x_campaign=x_campaign,
        y_campaign=y_campaign,
        x_forest_campaign=x_forest_campaign,
        y_forest_campaign=y_forest_campaign,
        x_nonforest_campaign=x_nonforest_campaign,
        y_nonforest_campaign=y_nonforest_campaign,
        x_filtered=x_filtered,
        y_filtered=y_filtered,
        x_forest_filtered=x_forest_filtered,
        y_forest_filtered=y_forest_filtered,
        x_nonforest_filtered=x_nonforest_filtered,
        y_nonforest_filtered=y_nonforest_filtered,
        x_prodes_scaled=x_prodes_scaled,
        x_test_prodes_scaled=x_test_prodes_scaled,
        x_campaign_scaled=x_campaign_scaled,
        x_test_campaign_scaled=x_test_campaign_scaled,
        x_forest_campaign_scaled=x_forest_campaign_scaled,
        x_nonforest_campaign_scaled=x_nonforest_campaign_scaled,
        x_filtered_scaled=x_filtered_scaled,
        x_test_filtered_scaled=x_test_filtered_scaled,
        x_forest_filtered_scaled=x_forest_filtered_scaled,
        x_nonforest_filtered_scaled=x_nonforest_filtered_scaled,
        df_test=df_test,
        test_segment_ids=compact_replicate_values(df_test, "Segment_id"),
    )
