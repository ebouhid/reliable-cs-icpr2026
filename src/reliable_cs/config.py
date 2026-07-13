"""Configuration and constants for the SVM reproducibility pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RANDOM_SEED = 42
DA_FLOW = "OVER_FILTERED_BY_MEDIAN_ENTROPY"
N_FOLDS = 5
SVM_C = 1.0
SVM_KERNEL = "linear"
GNI_NOISE_STD = 0.2

OUTLIER_DISPLAY_NAMES = [
    "With Outliers",
    "Perc. Task",
    "Perc. Global",
    "Tukey",
    "Z-Score",
    "MAD",
]

OUTLIER_FILE_SUFFIXES = [
    "with_outliers",
    "perc_task",
    "perc_global",
    "tukey",
    "zscore",
    "mad",
]

TRAIN_DROP_COLUMNS = [
    "Task_id",
    "Majority_response",
    "is_match",
    "Entropy",
    "Mean_duration_response",
]

# Feature columns start after Segment_id, Study_Area, PRODES_class, band
FEATURE_SLICE_START = 4

# Raw CSVs store one row per band; labels/entropy are taken from every 5th row
BANDS_PER_SEGMENT_IN_CSV = 5


@dataclass(frozen=True)
class CampaignConfig:
    key: str
    name: str
    path_name: str
    bands: list[int]
    train_files: list[str]
    test_file: str


def get_campaign(key: str, data_root: Path) -> CampaignConfig:
    key = key.lower()
    if key == "sentinel":
        prefix = "sentinel_big_campaign"
        return CampaignConfig(
            key="sentinel",
            name="Sentinel-2 Campaign",
            path_name="Sentinel-2_Campaign",
            bands=[11, 4, 3, 1],
            train_files=[
                str(data_root / "train" / f"{prefix}_with_outliers_train.csv"),
                str(data_root / "train" / f"{prefix}_percentile_by_task_train.csv"),
                str(data_root / "train" / f"{prefix}_percentile_global_train.csv"),
                str(data_root / "train" / f"{prefix}_tukey_train.csv"),
                str(data_root / "train" / f"{prefix}_zscore_train.csv"),
                str(data_root / "train" / f"{prefix}_mad_train.csv"),
            ],
            test_file=str(data_root / "test" / f"{prefix}_test.csv"),
        )
    if key == "landsat":
        prefix = "landsat_flip_campaign"
        return CampaignConfig(
            key="landsat",
            name="Landsat-8 Campaign",
            path_name="Landsat-8_Campaign",
            bands=[6, 4, 3, 1],
            train_files=[
                str(data_root / "train" / f"{prefix}_with_outliers_train.csv"),
                str(data_root / "train" / f"{prefix}_percentile_by_task_train.csv"),
                str(data_root / "train" / f"{prefix}_percentile_global_train.csv"),
                str(data_root / "train" / f"{prefix}_tukey_train.csv"),
                str(data_root / "train" / f"{prefix}_zscore_train.csv"),
                str(data_root / "train" / f"{prefix}_mad_train.csv"),
            ],
            test_file=str(data_root / "test" / f"{prefix}_test.csv"),
        )
    raise ValueError(f"Unknown campaign: {key!r}. Use 'landsat' or 'sentinel'.")
