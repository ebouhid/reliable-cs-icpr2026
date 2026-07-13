"""End-to-end campaign experiment orchestration."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from reliable_cs.config import DA_FLOW, RANDOM_SEED, get_campaign
from reliable_cs.export import save_classification_reports, save_metrics_table
from reliable_cs.metrics import build_metrics_table
from reliable_cs.plots import save_comparison_plots
from reliable_cs.preprocess import prepare_data
from reliable_cs.sampling import build_balanced_sets


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_campaign(
    campaign_key: str,
    repo_root: Path,
    seed: int = RANDOM_SEED,
) -> Path:
    """Run the full SVM experiment for one campaign and write all outputs."""
    set_seeds(seed)
    data_root = repo_root / "data"
    metrics_root = repo_root / "metrics_tables"
    reports_root = repo_root / "classification_reports"
    figures_root = repo_root / "figures"

    campaign = get_campaign(campaign_key, data_root)
    print(f"\n{'=' * 70}")
    print(f"Running campaign: {campaign.name}")
    print(f"Seed: {seed} | Flow: {DA_FLOW}")
    print(f"{'=' * 70}\n")

    data = prepare_data(campaign)
    balanced = build_balanced_sets(data, seed=seed)
    df_metrics, fitted = build_metrics_table(data, balanced, seed=seed)

    metrics_path = save_metrics_table(
        df_metrics, campaign.path_name, DA_FLOW, metrics_root
    )
    save_classification_reports(data, fitted, reports_root)
    save_comparison_plots(
        df_metrics, campaign.name, campaign.path_name, figures_root, da_flow=DA_FLOW
    )
    return metrics_path
