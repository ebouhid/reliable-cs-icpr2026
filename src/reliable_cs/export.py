"""Export metrics tables and classification report CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reliable_cs.config import OUTLIER_FILE_SUFFIXES


def save_metrics_table(
    df_metrics: pd.DataFrame,
    campaign_path_name: str,
    da_flow: str,
    metrics_root: Path,
) -> Path:
    out_dir = metrics_root / campaign_path_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"comprehensive_metrics_5fold_{campaign_path_name}_{da_flow}.csv"
    df_metrics.to_csv(path, index=False)
    print(f"Metrics table saved to: {path}")
    return path


def save_classification_reports(
    data,
    fitted: dict,
    reports_root: Path,
) -> Path:
    """Write per-method prediction CSVs (segment_id, features, labels)."""
    out_dir = reports_root / data.campaign.path_name
    out_dir.mkdir(parents=True, exist_ok=True)

    num_features = data.x_test_joined.shape[1]
    feature_columns = [f"Feature_{i}" for i in range(1, num_features + 1)]

    def _save(segment_ids, features, y_true, y_pred, model_name, outlier_name=None):
        df_report = pd.DataFrame(features, columns=feature_columns)
        df_report.insert(0, "segment_id", segment_ids)
        df_report["class_prodes"] = y_true
        df_report["predicted_class"] = y_pred
        if outlier_name:
            filename = out_dir / f"svm_{model_name}_{outlier_name}.csv"
        else:
            filename = out_dir / f"svm_{model_name}.csv"
        df_report.to_csv(filename, index=False)
        print(f"Saved: {filename}")

    # PRODES
    entry = fitted["prodes_based"]
    y_pred = entry["model"].predict(entry["X_test"])
    _save(data.test_segment_ids, entry["X_test"], data.y_test, y_pred, "prodes_based")

    per_outlier_methods = [
        "campaign_based",
        "campaign_based_median_entropy",
        "random_undersample",
        "ascending_entropy_undersample",
        "descending_entropy_undersample",
        "gni",
        "smote",
        "adasyn",
    ]
    for method_key in per_outlier_methods:
        for idx, outlier_name in enumerate(OUTLIER_FILE_SUFFIXES):
            entry = fitted[method_key][idx]
            y_pred = entry["model"].predict(entry["X_test"])
            _save(
                data.test_segment_ids,
                entry["X_test"],
                data.y_test,
                y_pred,
                method_key,
                outlier_name,
            )

    print(f'\n=== All classification reports saved to "{out_dir}/" ===')
    return out_dir
