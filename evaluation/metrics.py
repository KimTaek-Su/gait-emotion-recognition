from __future__ import annotations

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


def classification_summary(y_true, y_pred, labels=None) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if labels else confusion_matrix(y_true, y_pred).tolist(),
    }