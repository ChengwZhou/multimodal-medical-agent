# metrics.py
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    cohen_kappa_score, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score,
    precision_score, recall_score,
)


def compute_metrics(y_true, y_pred, average='macro', use_weighted=False,
                    y_score=None):
    """
    Compute classification metrics commonly used in biomedical datasets.

    Args
    ----
    y_true      : ground-truth labels  (list / np.ndarray / torch.Tensor)
    y_pred      : predicted labels     (list / np.ndarray / torch.Tensor)
    average     : F1/precision/recall averaging ('micro'|'macro'|'weighted')
    use_weighted: if True, forces weighted averaging (overrides `average`)
    y_score     : soft prediction probabilities [N, C] or [N] for binary —
                  required for AUC-ROC; skipped when None

    Returns
    -------
    dict with keys:
        accuracy            — overall accuracy
        balanced_accuracy   — mean per-class recall (good for imbalanced data)
        f1_score            — macro/weighted F1
        f1_per_class        — per-class F1 [n_classes]
        precision           — macro/weighted precision
        recall              — macro/weighted recall (= sensitivity)
        specificity_mean    — mean per-class specificity (TN/(TN+FP))
        cohen_kappa         — Cohen's κ (chance-corrected agreement)
        mcc                 — Matthews Correlation Coefficient
        auc_roc             — macro OvR AUC-ROC (requires y_score)
        confusion_matrix    — confusion matrix [n_classes, n_classes]
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1_avg = 'weighted' if use_weighted else average

    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, average=f1_avg, zero_division=0)
    f1_cls   = f1_score(y_true, y_pred, average=None,   zero_division=0)
    prec     = precision_score(y_true, y_pred, average=f1_avg, zero_division=0)
    rec      = recall_score(y_true, y_pred,    average=f1_avg, zero_division=0)
    kappa    = cohen_kappa_score(y_true, y_pred)
    mcc      = matthews_corrcoef(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred)

    # Per-class specificity: TN / (TN + FP)
    specificity = _per_class_specificity(cm)
    spec_mean   = float(np.mean(specificity))

    # AUC-ROC (macro OvR)
    auc = None
    if y_score is not None:
        try:
            classes = np.unique(y_true)
            multi   = len(classes) > 2
            auc = roc_auc_score(
                y_true, y_score,
                multi_class='ovr' if multi else 'raise',
                average='macro',
            )
        except Exception:
            auc = None

    return {
        "accuracy":          acc,
        "balanced_accuracy": bal_acc,
        "f1_score":          f1,
        "f1_per_class":      f1_cls,
        "precision":         prec,
        "recall":            rec,
        "specificity_mean":  spec_mean,
        "cohen_kappa":       kappa,
        "mcc":               mcc,
        "auc_roc":           auc,
        "confusion_matrix":  cm,
    }


def _per_class_specificity(cm: np.ndarray) -> np.ndarray:
    """Compute per-class specificity (TN/(TN+FP)) from a confusion matrix."""
    n = cm.shape[0]
    spec = np.zeros(n)
    total = cm.sum()
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        spec[i] = tn / denom if denom > 0 else 0.0
    return spec


def print_metrics(metrics):
    print("===val metrics===")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1-score:          {metrics['f1_score']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"Specificity(mean): {metrics['specificity_mean']:.4f}")
    print(f"Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")
    print(f"MCC:               {metrics['mcc']:.4f}")
    if metrics.get("auc_roc") is not None:
        print(f"AUC-ROC:           {metrics['auc_roc']:.4f}")
    if metrics.get("f1_per_class") is not None:
        cls_str = "  ".join(f"{v:.3f}" for v in metrics["f1_per_class"])
        print(f"F1 per class:      [{cls_str}]")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
