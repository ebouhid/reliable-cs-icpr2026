"""Data loading and multi-band feature joining."""

from __future__ import annotations

import pandas as pd

from reliable_cs.config import (
    BANDS_PER_SEGMENT_IN_CSV,
    FEATURE_SLICE_START,
    TRAIN_DROP_COLUMNS,
    CampaignConfig,
)


def join_features_band_per_segment(
    dataframe: pd.DataFrame,
    bands_list: list[int],
    columns_to_ignore: int = FEATURE_SLICE_START,
) -> pd.DataFrame:
    """Concatenate Haralick features across selected bands into one row per segment."""
    segment_features_join_bands_list: list = []
    global_list: list = []
    bands_len = len(bands_list)
    bands_count = 1

    for _, row in dataframe.iterrows():
        if bands_count <= bands_len:
            if row["band"] in bands_list:
                segment_features_join_bands_list.extend(list(row[columns_to_ignore:]))
                bands_count += 1
        else:
            bands_count = 2
            global_list.append(segment_features_join_bands_list)
            if row["band"] in bands_list:
                segment_features_join_bands_list = list(row[columns_to_ignore:])
            else:
                segment_features_join_bands_list = []

    global_list.append(segment_features_join_bands_list)
    return pd.DataFrame(global_list)


def compact_replicate_values(dataframe: pd.DataFrame, column: str) -> list:
    """Keep one value per segment (last band row among the fixed CSV band count)."""
    count_bands = 1
    compact_values_list = []
    for _, row in dataframe.iterrows():
        if count_bands < BANDS_PER_SEGMENT_IN_CSV:
            count_bands += 1
            continue
        compact_values_list.append(row[column])
        count_bands = 1
    return compact_values_list


def standardize_y(dataframe: pd.DataFrame, data_type: str) -> tuple[list, list]:
    """Extract one label per segment from multi-band CSV rows."""
    count_bands = 1
    y_list_prodes: list = []
    y_list_majority: list = []

    for _, row in dataframe.iterrows():
        if count_bands < BANDS_PER_SEGMENT_IN_CSV:
            count_bands += 1
            continue
        if data_type == "train":
            y_list_prodes.append(row["PRODES_class"])
            y_list_majority.append(row["Majority_response"])
        else:
            y_list_prodes.append(row["PRODES_class"])
        count_bands = 1

    return y_list_prodes, y_list_majority


def load_campaign_tables(campaign: CampaignConfig) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    """Load train CSVs (all outlier variants) and the shared test CSV."""
    df_train = [pd.read_csv(path) for path in campaign.train_files]
    x_train = [df.drop(columns=TRAIN_DROP_COLUMNS) for df in df_train]
    df_test = pd.read_csv(campaign.test_file)
    return df_train, x_train, df_test
