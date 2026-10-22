import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score

from src.classify.data import NUM_CLASSES

LABELS = list(range(NUM_CLASSES))
METRIC_COLS = ["macro_f1", "qwk", "acc"] + [f"recall_{k}" for k in LABELS]


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rec = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if len(np.unique(y_true)) > 1 else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)),
    }
    out.update({f"recall_{k}": float(rec[k]) for k in LABELS})
    return out


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=LABELS)
