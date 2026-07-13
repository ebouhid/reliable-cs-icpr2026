"""Metric computation and comprehensive results table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from reliable_cs.config import OUTLIER_DISPLAY_NAMES
from reliable_cs.train import make_cv, make_svm


def compute_label_agreement(prodes_labels, campaign_labels):
    """Return global / PRODES-Forest / PRODES-Non-forest mismatch percentages."""
    prodes_list = list(prodes_labels) if hasattr(prodes_labels, "tolist") else list(prodes_labels)
    campaign_list = list(campaign_labels)

    valid_indices = [i for i, label in enumerate(campaign_list) if label in ["Forest", "Non-forest"]]
    if not valid_indices:
        return 0.0, 0.0, 0.0

    prodes_valid = [prodes_list[i] for i in valid_indices]
    campaign_valid = [campaign_list[i] for i in valid_indices]

    n_samples = len(prodes_valid)
    mismatches = sum(1 for p, c in zip(prodes_valid, campaign_valid) if p != c)
    global_mismatch_pct = 100 * mismatches / n_samples if n_samples > 0 else 0

    prodes_forest_count = sum(1 for p in prodes_valid if p == "Forest")
    prodes_forest_campaign_nonforest = sum(
        1 for p, c in zip(prodes_valid, campaign_valid) if p == "Forest" and c == "Non-forest"
    )
    prodes_forest_mismatch_pct = (
        100 * prodes_forest_campaign_nonforest / prodes_forest_count if prodes_forest_count > 0 else 0
    )

    prodes_nonforest_count = sum(1 for p in prodes_valid if p == "Non-forest")
    prodes_nonforest_campaign_forest = sum(
        1 for p, c in zip(prodes_valid, campaign_valid) if p == "Non-forest" and c == "Forest"
    )
    prodes_nonforest_mismatch_pct = (
        100 * prodes_nonforest_campaign_forest / prodes_nonforest_count
        if prodes_nonforest_count > 0
        else 0
    )

    return global_mismatch_pct, prodes_forest_mismatch_pct, prodes_nonforest_mismatch_pct


def compute_metrics(y_true, y_pred, pos_label: str = "Non-forest"):
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    labels = ["Forest", "Non-forest"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return bal_acc, f1, precision, recall, sensitivity, specificity


def count_samples(y, pos_label: str = "Non-forest", neg_label: str = "Forest"):
    y_arr = np.array(y)
    return int(np.sum(y_arr == pos_label)), int(np.sum(y_arr == neg_label))


def compute_5fold_metrics(model: SVC, cv: StratifiedKFold, X_train, y_train, X_test, y_test):
    train_metrics = {
        "bal_acc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "sensitivity": [],
        "specificity": [],
    }
    test_metrics = {
        "bal_acc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "sensitivity": [],
        "specificity": [],
    }

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]

        model.fit(X_tr, y_tr)

        y_val_pred = model.predict(X_val)
        bal_acc, f1, prec, rec, sens, spec = compute_metrics(y_val, y_val_pred)
        train_metrics["bal_acc"].append(bal_acc)
        train_metrics["f1"].append(f1)
        train_metrics["precision"].append(prec)
        train_metrics["recall"].append(rec)
        train_metrics["sensitivity"].append(sens)
        train_metrics["specificity"].append(spec)

        y_test_pred = model.predict(X_test)
        bal_acc, f1, prec, rec, sens, spec = compute_metrics(y_test, y_test_pred)
        test_metrics["bal_acc"].append(bal_acc)
        test_metrics["f1"].append(f1)
        test_metrics["precision"].append(prec)
        test_metrics["recall"].append(rec)
        test_metrics["sensitivity"].append(sens)
        test_metrics["specificity"].append(spec)

    return train_metrics, test_metrics


def format_mean_std(values) -> str:
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def _row(
    method: str,
    outlier: str,
    train_pos: int,
    train_neg: int,
    test_pos: int,
    test_neg: int,
    mismatch,
    train_m: dict,
    test_m: dict,
) -> dict:
    if mismatch is None:
        label_m, pf_m, pnf_m = "N/A", "N/A", "N/A"
    else:
        global_m, pf, pnf = mismatch
        label_m, pf_m, pnf_m = f"{global_m:.1f}", f"{pf:.1f}", f"{pnf:.1f}"

    return {
        "Method": method,
        "Outlier Approach": outlier,
        "Train Pos": train_pos,
        "Train Neg": train_neg,
        "Test Pos": test_pos,
        "Test Neg": test_neg,
        "Label Mismatch (%)": label_m,
        "PRODES-F Mismatch (%)": pf_m,
        "PRODES-NF Mismatch (%)": pnf_m,
        "Train Bal. Acc.": format_mean_std(train_m["bal_acc"]),
        "Train F1": format_mean_std(train_m["f1"]),
        "Train Precision": format_mean_std(train_m["precision"]),
        "Train Recall": format_mean_std(train_m["recall"]),
        "Test Bal. Acc.": format_mean_std(test_m["bal_acc"]),
        "Test F1": format_mean_std(test_m["f1"]),
        "Test Precision": format_mean_std(test_m["precision"]),
        "Test Recall": format_mean_std(test_m["recall"]),
        "Test Sensitivity": format_mean_std(test_m["sensitivity"]),
        "Test Specificity": format_mean_std(test_m["specificity"]),
    }


def build_metrics_table(data, balanced: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """
    Run 5-fold evaluation for all methods.

    Returns the metrics DataFrame and a dict of fitted models (left in last-fold
    state, matching the notebook) plus their corresponding test feature matrices
    for classification-report export.
    """
    results = []
    fitted = {}
    test_pos, test_neg = count_samples(data.y_test)
    n = len(data.campaign.train_files)

    print("Computing 5-fold CV metrics for all methods...")
    print("=" * 60)

    # 1. PRODES-based
    print("Processing: PRODES-based...")
    model = make_svm(seed)
    cv = make_cv(seed)
    train_m, test_m = compute_5fold_metrics(
        model, cv, data.x_prodes_scaled, data.y_prodes, data.x_test_prodes_scaled, data.y_test
    )
    train_pos, train_neg = count_samples(data.y_prodes)
    results.append(
        _row("PRODES-based", "N/A", train_pos, train_neg, test_pos, test_neg, None, train_m, test_m)
    )
    fitted["prodes_based"] = {
        "model": model,
        "X_test": data.x_test_prodes_scaled,
        "outlier": None,
    }

    method_specs = [
        (
            "Campaign-based",
            "campaign_based",
            data.x_campaign_scaled,
            data.y_campaign,
            data.x_test_campaign_scaled,
        ),
        (
            "Campaign + Median Entropy",
            "campaign_based_median_entropy",
            data.x_filtered_scaled,
            data.y_filtered,
            data.x_test_filtered_scaled,
        ),
        (
            "Random Undersample",
            "random_undersample",
            balanced["random_undersample"][0],
            balanced["random_undersample"][1],
            data.x_test_filtered_scaled,
        ),
        (
            "Ascending Entropy Undersample",
            "ascending_entropy_undersample",
            balanced["ascending_entropy_undersample"][0],
            balanced["ascending_entropy_undersample"][1],
            data.x_test_filtered_scaled,
        ),
        (
            "Descending Entropy Undersample",
            "descending_entropy_undersample",
            balanced["descending_entropy_undersample"][0],
            balanced["descending_entropy_undersample"][1],
            data.x_test_filtered_scaled,
        ),
        (
            "GNI",
            "gni",
            balanced["gni"][0],
            balanced["gni"][1],
            data.x_test_filtered_scaled,
        ),
        (
            "SMOTE",
            "smote",
            balanced["smote"][0],
            balanced["smote"][1],
            data.x_test_filtered_scaled,
        ),
        (
            "ADASYN",
            "adasyn",
            balanced["adasyn"][0],
            balanced["adasyn"][1],
            data.x_test_filtered_scaled,
        ),
    ]

    for method_name, method_key, x_trains, y_trains, x_tests in method_specs:
        fitted[method_key] = []
        for idx, outlier_name in enumerate(OUTLIER_DISPLAY_NAMES):
            print(f"Processing: {method_name} - {outlier_name}...")
            model = make_svm(seed)
            cv = make_cv(seed)
            train_m, test_m = compute_5fold_metrics(
                model, cv, x_trains[idx], y_trains[idx], x_tests[idx], data.y_test
            )
            train_pos, train_neg = count_samples(y_trains[idx])
            mismatch = compute_label_agreement(data.y_prodes, data.y_majority_raw[idx])
            results.append(
                _row(
                    method_name,
                    outlier_name,
                    train_pos,
                    train_neg,
                    test_pos,
                    test_neg,
                    mismatch,
                    train_m,
                    test_m,
                )
            )
            fitted[method_key].append(
                {"model": model, "X_test": x_tests[idx], "outlier_idx": idx}
            )

    df_metrics = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("=== Comprehensive Metrics Table (Mean ± Std of 5 Folds) ===")
    print(f"=== Campaign: {data.campaign.name} ===")
    print(f"Total method-outlier combinations evaluated: {len(df_metrics)}")
    print("=" * 60)
    return df_metrics, fitted
