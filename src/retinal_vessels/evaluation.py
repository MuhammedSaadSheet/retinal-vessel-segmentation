import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_segmentation(pred_mask, gt_mask, fov_mask=None):
    """
    Evaluate a predicted vessel mask against a ground-truth mask.

    Parameters:
        pred_mask (numpy.ndarray): Predicted binary vessel mask.
        gt_mask (numpy.ndarray): Ground-truth binary vessel mask.
        fov_mask (numpy.ndarray, optional): Field-of-view mask.
            If provided, evaluation is performed only inside the visible
            retinal area.

    Returns:
        dict: Dictionary containing segmentation metrics.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if fov_mask is not None:
        fov = fov_mask.astype(bool)
        pred = pred[fov]
        gt = gt[fov]
    else:
        pred = pred.ravel()
        gt = gt.ravel()

    tn, fp, fn, tp = confusion_matrix(
        gt,
        pred,
        labels=[0, 1],
    ).ravel()

    eps = 1e-8

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "dice": dice,
        "iou": iou,
    }


def print_metrics(metrics):
    """
    Print evaluation metrics in a readable format.

    Parameters:
        metrics (dict): Dictionary returned by evaluate_segmentation().
    """
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")