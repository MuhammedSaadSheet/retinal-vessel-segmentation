import numpy as np

from retinal_vessels.evaluation import evaluate_segmentation


def test_evaluate_segmentation_basic_case():
    """
    Test evaluation metrics on a small known example.

    Ground truth:
        2 vessel pixels
        2 background pixels

    Prediction:
        1 vessel correctly detected
        1 vessel missed
        2 background pixels correctly detected

    Expected:
        TP = 1
        TN = 2
        FP = 0
        FN = 1
    """
    gt_mask = np.array(
        [
            [1, 0],
            [1, 0],
        ]
    )

    pred_mask = np.array(
        [
            [1, 0],
            [0, 0],
        ]
    )

    metrics = evaluate_segmentation(pred_mask, gt_mask)

    assert np.isclose(metrics["accuracy"], 0.75)
    assert np.isclose(metrics["sensitivity"], 0.5)
    assert np.isclose(metrics["specificity"], 1.0)
    assert np.isclose(metrics["precision"], 1.0)
    assert np.isclose(metrics["dice"], 2 / 3)
    assert np.isclose(metrics["iou"], 0.5)


def test_evaluate_segmentation_with_fov_mask():
    """
    Test whether the field-of-view mask limits the evaluation area.

    Only pixels where fov_mask == 1 should be included in the metric calculation.
    """
    gt_mask = np.array(
        [
            [1, 0],
            [1, 0],
        ]
    )

    pred_mask = np.array(
        [
            [1, 1],
            [0, 0],
        ]
    )

    fov_mask = np.array(
        [
            [1, 0],
            [1, 0],
        ]
    )

    metrics = evaluate_segmentation(pred_mask, gt_mask, fov_mask=fov_mask)

    assert np.isclose(metrics["accuracy"], 0.5)
    assert np.isclose(metrics["sensitivity"], 0.5)
    assert np.isclose(metrics["dice"], 2 / 3)