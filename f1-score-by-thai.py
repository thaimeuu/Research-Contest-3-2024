import os
import numpy as np
import cv2
from combine_pred_true import combine_pred_true


def f1_by_thai(y_true_13: np.array, y_true_1: np.array, y_pred: np.array, return_precision=True, return_recall=True) -> float:
    """
    ultimate f1-score by thai ^_^

    Args:
        y_true_13 (np.array): any size allowed (Each junction takes up 13 pixels (i.e., cv2.circle(radius=2)) (important))
        y_true_1 (np.array): any size allowed (Each junction takes up 1 pixels (important)
        y_pred (np.array): any size allowed (Each junction takes up 1 pixel)
        return_precision (bool): Defaults:True
        return_recall (bool): Defaults:True

    Returns:
        f1: float: f1 score
    """
    true_13, true_1, pred = np.copy(y_true_13), np.copy(y_true_1), np.copy(y_pred)

    junction_true_13 = np.argwhere(true_13 > 0)
    junction_true_1 = np.argwhere(true_1 > 0)
    junction_pred = np.argwhere(pred > 0)

    true_13 = tuple(map(tuple, junction_true_13))
    n_true_clusters = len(junction_true_1)
    pred = tuple(map(tuple, junction_pred))

    true_pos, false_pos, false_neg = (0, 0, 0)

    for junction in pred:
        if junction in true_13:
            true_pos += 1
        else:
            false_pos += 1

    false_neg = n_true_clusters - true_pos

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score if not (return_precision and return_recall) else [f1_score, precision, recall] if (return_precision and return_recall) else [f1_score, precision] if (return_precision and not return_recall) else [f1_score, recall]


if __name__ == "__main__":
    y_true_13 = cv2.imread("y_true/crossing_number/y_true_13/1C8-7-1.png", cv2.IMREAD_GRAYSCALE)
    y_true_1 = cv2.imread("y_true/crossing_number/y_true_1/1C8-7-1.png", cv2.IMREAD_GRAYSCALE)
    y_pred = cv2.imread("y_pred/crossing-number/Zhang-Suen/RUC-Net/1C8-7-1.png", cv2.IMREAD_GRAYSCALE)
    f1_score, precision, recall = f1_by_thai(y_true_13, y_true_1, y_pred)
    print(f1_score, precision, recall)
    
    combine_pred_true(y_true_13, y_pred, visualization=True)
    