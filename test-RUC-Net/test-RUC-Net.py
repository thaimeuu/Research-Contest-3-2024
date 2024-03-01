import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, f1_score

y_pred = cv2.imread("test-RUC-Net/y_pred_1_1C8-12-3.png", cv2.IMREAD_GRAYSCALE)
y_true = cv2.imread("test-RUC-Net/y_true_1_1C8-12-3.png", cv2.IMREAD_GRAYSCALE)

y_pred = y_pred.flatten()
y_true = y_true.flatten()

white_px_y_pred = np.argwhere(y_pred == 1)
white_px_y_true = np.argwhere(y_true == 1)

print(f"y_pred: {y_pred.shape}, unique values: {np.unique(y_pred, return_counts=True)}")
print(f"y_true: {y_true.shape}, unique values: {np.unique(y_true, return_counts=True)}")

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(cm)
print(f"True negatives: {cm[0, 0]}\nFalse negatives: {cm[1, 0]}\nTrue positives: {cm[1, 1]}\nFalse positives: {cm[0, 1]}")
print(f"True positives at: {np.intersect1d(white_px_y_pred, white_px_y_true)}")
f1 = f1_score(y_true, y_pred, average='binary')
print(f"F1-score: {f1}")
