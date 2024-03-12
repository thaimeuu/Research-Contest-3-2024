from combine_pred_true import combine_pred_true
import numpy as np
import cv2
from sklearn.metrics import f1_score
from new_coordinate import new_coordinate


# y_pred_path = "dataset_1_y_pred/crossing-number/gradient-based-optimization/RUC-Net/1C8-9-3.png"
y_pred_path = "dataset_1_y_pred/crossing-number/Zhang-Suen/RUC-Net/1C8-9-3.png"
y_pred = cv2.imread(y_pred_path, cv2.IMREAD_GRAYSCALE)

y_true_path = "dataset_1_y_true_main_axis/crossing_number/y_true_25/1C8-9-3.png"
y_true = cv2.imread(y_true_path, cv2.IMREAD_GRAYSCALE)

junction_true_25 = np.argwhere(y_true > 0)
    
lower_bound = np.min(junction_true_25, axis=0)[0]
upper_bound = np.max(junction_true_25, axis=0)[0]
left_bound = np.min(junction_true_25, axis=0)[1]
right_bound = np.max(junction_true_25, axis=0)[1]

y_pred[:lower_bound - 1, :] = 0  # -1 for loose bound
y_pred[upper_bound + 1 + 1:, :] = 0  # + 1 for loose bound
# y_pred[:, :left_bound - 1] = 0  # -1 for loose bound
y_pred[:, right_bound + 1 + 1:] = 0  # + 1 for loose bound

# cv2.imwrite(f"dataset_1_y_pred_main_axis/zhang-cn/{y_pred_path[-11:]}", y_pred)

white_px = np.argwhere(y_pred > 0)

original_grayscale = cv2.imread("dataset_1_grayscale/1C8-9-3.JPG")
original_grayscale = cv2.resize(original_grayscale, [256, 256])

for x, y in white_px:
    cv2.circle(original_grayscale, [y, x], 1, (0, 0, 255), -1)
    
    
cv2.imwrite("haha.png", original_grayscale)

cv2.imshow("aks", original_grayscale)
cv2.waitKey(0)
cv2.destroyAllWindows()
