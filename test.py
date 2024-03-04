from combine_pred_true import combine_pred_true
import numpy as np
import cv2
from sklearn.metrics import f1_score


y_pred = cv2.imread("test-img/y_pred.png", cv2.IMREAD_GRAYSCALE)
y_true = cv2.imread("test-img/y_true.png", cv2.IMREAD_GRAYSCALE)

print(np.unique(y_pred, return_counts=True))
print(np.unique(y_true, return_counts=True))

img = combine_pred_true(y_true, y_pred, save_path="D:/HRG/SVNCKH 3-2024/test-img/combined.png")

y_pred[y_pred == 255] = 1
y_true[y_true == 255] = 1

y_pred = y_pred.flatten()
y_true = y_true.flatten()

f1 = f1_score(y_true, y_pred)
print(f1)
