import numpy as np
import cv2
from sklearn.metrics import confusion_matrix


def Accuracy12(GT: np.array, seg: np.array, beta: float = 0.3) -> list[float]:
    """
    fbeta-score by Le Cong Hieu

    Args:
        GT (ndarray): any size is accepted
        seg (ndarray): any size is accepted
        beta (float): beta. Defaults=0.3

    Returns:
        F, precision, recall
    """
    r = []
    p = []
    F = 0
    GT1 = GT.copy()
    seg1 = seg.copy()

    # [x,y] = np.argwhere(GT >0)
    GT1[GT1 > 0] = 1
    GT1 = np.ndarray.flatten(GT1)
    # [x,y] = np.argwhere(seg > 0)
    seg1[seg1 > 0] = 1
    seg1 = np.ndarray.flatten(seg1)

    CM = confusion_matrix(GT1, seg1)
    c = np.shape(CM)
    for i in range(c[1]):
        if np.sum(CM[i, :]) == 0:
            r.append(0)
        else:
            a = CM[i, i] / (np.sum(CM[i, :]))
            r.append(a)
        if np.sum(CM[:, i]) == 0:
            p.append(0)
        else:
            p.append(CM[i, i] / (np.sum(CM[:, i])))
    print(CM, np.sum(CM))
    F = (1 + beta) * (np.mean(r) * np.mean(p)) / (beta * np.mean(p) + np.mean(r))
    return F, np.mean(p), np.mean(r)


def f_beta(y_true: np.array, y_pred: np.array, beta: float = 0.3) -> list[float]:
    """
    fbeta-score by Thai

    Args:
        y_true (np.array): y_true of any size
        y_pred (np.array): y_pred of any size
        beta (float, optional): beta. Defaults to 0.3.

    Returns:
        fbeta, precision, recall
    """
    y_true[y_true == 255] = 1
    y_pred[y_pred == 255] = 1
    y_true, y_pred = y_true.flatten(), y_pred.flatten()

    cm = confusion_matrix(y_true, y_pred)
    tn, fn, fp, tp = cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]
    precision = np.mean([tn / (tn + fn), tp / (tp + fp)])
    recall = np.mean([tn / (tn + fp), tp / (tp + fn)])
    # fbeta = (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + fp + beta ** 2 * fn)
    fbeta = (1 + beta) * precision * recall / (beta * precision + recall)
    print(cm, np.sum(cm))
    return fbeta, precision, recall


if __name__ == "__main__":
    y_true = cv2.imread("test-RUC-Net/y_true_255_1C8-12-3.png", cv2.IMREAD_GRAYSCALE)
    y_pred = cv2.imread("test-RUC-Net/y_pred_255_1C8-12-3.png", cv2.IMREAD_GRAYSCALE)

    fbeta, precision, recall = Accuracy12(y_true, y_pred, 0.3)
    print(fbeta, precision, recall)

    fbeta, precision, recall = f_beta(y_true.copy(), y_pred.copy())
    print(fbeta, precision, recall)
