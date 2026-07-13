"""Comparison plots for balanced accuracy across methods."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reliable_cs.config import DA_FLOW, OUTLIER_DISPLAY_NAMES


def extract_mean(value_str: str) -> float:
    return float(value_str.split(" ± ")[0])


def _scores_for_method(df_metrics: pd.DataFrame, method: str) -> list[float]:
    return [
        extract_mean(
            df_metrics[
                (df_metrics["Method"] == method)
                & (df_metrics["Outlier Approach"] == outlier)
            ]["Test Bal. Acc."].values[0]
        )
        for outlier in OUTLIER_DISPLAY_NAMES
    ]


def save_comparison_plots(
    df_metrics: pd.DataFrame,
    campaign_name: str,
    campaign_path_name: str,
    figures_root: Path,
    da_flow: str = DA_FLOW,
) -> tuple[Path, Path]:
    out_dir = figures_root / campaign_path_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prodes_bal_acc = extract_mean(
        df_metrics[df_metrics["Method"] == "PRODES-based"]["Test Bal. Acc."].values[0]
    )
    prodes_scores = [prodes_bal_acc] * len(OUTLIER_DISPLAY_NAMES)
    campaign_scores = _scores_for_method(df_metrics, "Campaign-based")
    median_entropy_scores = _scores_for_method(df_metrics, "Campaign + Median Entropy")

    x = np.arange(len(OUTLIER_DISPLAY_NAMES))

    # Unbalanced approaches
    plt.figure(figsize=(10, 6))
    plt.plot(x, prodes_scores, label="Prodes-based", color="red")
    plt.plot(
        x,
        [campaign_scores[0]] * len(x),
        label="With Outliers\n(Campaign-based - Unbalanced)",
        color="green",
    )
    plt.scatter(
        x,
        [0] + campaign_scores[1:],
        label="Outlier Detection Methods\n(Campaign-based - Unbalanced)",
        marker="s",
        color="blue",
    )
    plt.scatter(
        x,
        median_entropy_scores,
        label="Median Entropy Filter\n(Unbalanced)",
        marker="p",
        color="orange",
    )
    plt.xticks(x, OUTLIER_DISPLAY_NAMES, rotation=45)
    plt.ylabel("Balanced Accuracy")
    plt.ylim(0.7, 1)
    plt.title(f"FGCS - Train Approaches - {campaign_name}")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    unbalanced_path = out_dir / "unbalanced_approaches.png"
    plt.savefig(unbalanced_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Balanced vs unbalanced
    random_undersample_scores = _scores_for_method(df_metrics, "Random Undersample")
    ascending_scores = _scores_for_method(df_metrics, "Ascending Entropy Undersample")
    descending_scores = _scores_for_method(df_metrics, "Descending Entropy Undersample")
    gni_scores = _scores_for_method(df_metrics, "GNI")
    smote_scores = _scores_for_method(df_metrics, "SMOTE")

    plt.figure(figsize=(10, 6))
    plt.plot(x, prodes_scores, label="Prodes-based", color="red")
    plt.plot(x, [campaign_scores[0]] * len(x), label="Campaign, With Outliers", color="green")
    plt.scatter(
        x,
        [0] + campaign_scores[1:],
        label="Outlier Detection Methods\n(Campaign-based - Unbalanced)",
        marker="o",
        color="gray",
    )
    plt.scatter(
        x,
        median_entropy_scores,
        label="Median Entropy Filter\n(Unbalanced)",
        marker="o",
        color="cyan",
    )
    plt.scatter(x, random_undersample_scores, label="Random Undersample", marker="D", color="purple")
    plt.scatter(
        x, ascending_scores, label="Ascending Entropy Undersample", marker="D", color="orange"
    )
    plt.scatter(
        x, descending_scores, label="Descending Entropy Undersample", marker="D", color="blue"
    )
    plt.scatter(x, gni_scores, label="Oversampling with GNI", marker="*", color="magenta")
    plt.scatter(x, smote_scores, label="Oversampling with SMOTE", marker="*", color="yellow")

    if da_flow == "OVER_FILTERED_BY_MEDIAN_ENTROPY":
        adasyn_scores = _scores_for_method(df_metrics, "ADASYN")
        plt.scatter(x, adasyn_scores, label="Oversampling with Adasyn", marker="*", color="black")

    plt.xticks(x, OUTLIER_DISPLAY_NAMES, rotation=45)
    plt.xlabel("Outlier Detection Method")
    plt.ylabel("Balanced Accuracy")
    plt.ylim(0.6, 1)
    if da_flow == "OVER_ORIGINAL_CAMPAIGN_DATA":
        plt.title(f"Test Balanced Accuracy by Outlier Detection Method - {campaign_name}")
    else:
        plt.title(
            f"Test Balanced Accuracy by Outlier Detection Method (Entropy-Filtered) - {campaign_name}"
        )
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    balanced_path = out_dir / "balanced_vs_unbalanced.png"
    plt.savefig(balanced_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figures saved to: {out_dir}/")
    return unbalanced_path, balanced_path
