# reliable-cs-icpr2026

Companion repository for the "Reliability-Aware Citizen Science for Environmental Machine Learning" ICPR 2026 paper on reliable citizen-science labels for deforestation mapping. It reproduces the linear SVM experiments that compare PRODES reference labels against campaign majority votes under several outlier filters and class-balancing strategies.

## What this repo reproduces

For each campaign (**Landsat-8** and **Sentinel-2**), the pipeline:

1. Joins multi-band Haralick texture features into one sample per segment
2. Removes Draw/Undefined campaign labels and optionally keeps samples with entropy ≤ median
3. Trains a linear SVM (`C=1.0`) with stratified 5-fold cross-validation (`random_state=42`)
4. Evaluates PRODES-based, campaign-based, median-entropy, undersampling, and oversampling (GNI, SMOTE, ADASYN) variants across six outlier-handling train sets
5. Writes metrics tables, classification report CSVs, and comparison figures

Correlation / Spearman analyses present in the archival notebooks are **not** part of this scripted workflow.

## Requirements

- [Pixi](https://pixi.sh/) (environment and task runner)
- Dataset from Hugging Face: [ebouhid/reliable-cs-icpr2026](https://huggingface.co/datasets/ebouhid/reliable-cs-icpr2026)

## Dataset

Download the train/test CSVs into `data/` before running experiments:

```bash
# with the Hugging Face CLI (https://huggingface.co/docs/huggingface_hub)
hf download ebouhid/reliable-cs-icpr2026 --repo-type dataset --local-dir data
```

Alternatively, browse and download files from the [dataset page](https://huggingface.co/datasets/ebouhid/reliable-cs-icpr2026). The pipeline expects:

```
data/train/*.csv
data/test/*.csv
```

## Quick start

```bash
pixi install
hf download ebouhid/reliable-cs-icpr2026 --repo-type dataset --local-dir data
pixi run reproduce
```

Individual campaigns:

```bash
pixi run run-landsat
pixi run run-sentinel
```

Or via the module CLI:

```bash
pixi run python -m reliable_cs --campaign all
pixi run python -m reliable_cs --campaign landsat --seed 42
```

## Outputs

| Path | Description |
|------|-------------|
| `metrics_tables/{Campaign}/comprehensive_metrics_5fold_*_OVER_FILTERED_BY_MEDIAN_ENTROPY.csv` | Mean ± std metrics over 5 folds |
| `classification_reports/{Campaign}/svm_*.csv` | Per-segment predictions (gitignored) |
| `figures/{Campaign}/unbalanced_approaches.png` | Unbalanced method comparison |
| `figures/{Campaign}/balanced_vs_unbalanced.png` | Full method comparison |

## Experiment settings

| Setting | Value |
|---------|--------|
| Random seed | `42` |
| SVM | Linear kernel, `C=1.0` |
| CV | Stratified 5-fold, shuffled |
| Balancing flow | `OVER_FILTERED_BY_MEDIAN_ENTROPY` |
| Landsat bands | 6, 4, 3, 1 |
| Sentinel bands | 11, 4, 3, 1 |
| Outlier train variants | with outliers, perc. task, perc. global, Tukey, Z-score, MAD |

## Repository layout

```
data/                  # train/test CSVs (from Hugging Face)
src/reliable_cs/       # reproducible Python package
notebooks/             # original notebooks (archival; not modified)
metrics_tables/        # regenerated metrics (tracked)
classification_reports/
figures/
pyproject.toml         # package + Pixi environment/tasks
```

The notebooks under `notebooks/` are the original exploratory sources. Prefer `pixi run reproduce` for paper results.

## License / citation

If you use this code or data, please cite the associated ICPR 2026 paper. (BibTex pending...)
