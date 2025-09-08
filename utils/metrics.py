# metrics.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred, average='macro'):
    """
    Accuracy, F1-score, Confusion Matrix

    Args:
        y_true (list or np.ndarray or torch.Tensor)
        y_pred (list or np.ndarray or torch.Tensor)
        average (str): F1-score 的计算方式 (micro, macro, weighted)

    Returns:
        dict: accuracy, f1, confusion_matrix
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def print_metrics(metrics):
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
