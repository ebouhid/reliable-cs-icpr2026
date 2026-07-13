"""SVM model construction and 5-fold helpers."""

from __future__ import annotations

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from reliable_cs.config import N_FOLDS, RANDOM_SEED, SVM_C, SVM_KERNEL


def make_svm(random_state: int = RANDOM_SEED) -> SVC:
    return SVC(C=SVM_C, kernel=SVM_KERNEL, random_state=random_state, verbose=False)


def make_cv(random_state: int = RANDOM_SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
