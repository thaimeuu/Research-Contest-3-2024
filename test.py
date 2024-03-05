from combine_pred_true import combine_pred_true
import numpy as np
import cv2
from sklearn.metrics import f1_score
from new_coordinate import new_coordinate


y_pred = cv2.imread("F1_score/crossing-number/Zhang-Suen/RUC-Net/1C8-9-2.png", cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread("Jouannic_dataset/1C8-9-2.JPG")
img_out = np.copy(original_image)
print(np.unique(y_pred, return_counts=True))

y_pred = cv2.resize(y_pred, [img_out.shape[1], img_out.shape[0]], interpolation=cv2.INTER_NEAREST)

cv2.imwrite("test-img/new_y_pred.png", y_pred)

print(np.unique(y_pred, return_counts=True))
white_px = np.argwhere(y_pred > 0)

for x, y in white_px:
    img_out[x, y] = (0, 0, 255)
    
cv2.imwrite("test-img/pred_junctions_on_original_img.png", img_out)
# y_true = cv2.imread("test-img/y_true.png", cv2.IMREAD_GRAYSCALE)

# print(np.unique(y_true, return_counts=True))

# img = combine_pred_true(y_true, y_pred, save_path="D:/HRG/SVNCKH 3-2024/test-img/combined.png")

# y_pred[y_pred == 255] = 1
# y_true[y_true == 255] = 1

# y_pred = y_pred.flatten()
# y_true = y_true.flatten()

# f1 = f1_score(y_true, y_pred)
# print(f1)
