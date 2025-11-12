# metrics.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred, average='macro', use_weighted=False):
    """
    Accuracy, F1-score, Confusion Matrix

    Args:
        y_true (list or np.ndarray or torch.Tensor)
        y_pred (list or np.ndarray or torch.Tensor)
        average (str): F1-score 的计算方式 (micro, macro, weighted)
        use_weighted (bool): 是否强制使用 weighted-F1
                             为 True 时，average 参数被忽略，自动改为 'weighted'

    Returns:
        dict: accuracy, f1_score, confusion_matrix
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # decide whether to use weighted
    f1_avg = 'weighted' if use_weighted else average

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=f1_avg)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def print_metrics(metrics):
    print("===val metrics===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


