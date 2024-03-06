from combine_pred_true import combine_pred_true
import numpy as np
import cv2
from sklearn.metrics import f1_score
from new_coordinate import new_coordinate


y_pred = cv2.imread("test-RUC-Net/y_pred_255_1C8-12-3.png", cv2.IMREAD_GRAYSCALE)
y_true = cv2.imread("test-RUC-Net/y_true_255_1C8-12-3.png", cv2.IMREAD_GRAYSCALE)

junction_true = np.argwhere(y_true)
junction_pred = np.argwhere(y_pred)
print(len(junction_pred))
print(len(junction_true))

for [x, y] in junction_true:
    # cv2.circle(y_true, [y, x], 2, 255, -1)  # one junction equals 13 pixels
    y_true[x-2:x+3, y-2:y+3] = 255  # one junction equals 25 pixels
    
junction_true = np.argwhere(y_true)
print(len(junction_true))

original_image = cv2.imread("grayscale_dataset/1C8-12-3.JPG")
original_image = cv2.resize(original_image, [256, 256])

for x, y in junction_true:
    original_image[x, y] = (255, 0, 0)
    
for x, y in junction_pred:
    original_image[x, y] = (0, 0, 255)
    
# cv2.imshow("original", original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite("test-RUC-Net/original_image_with_big_true_junctions.png", original_image)
    
# cv2.imshow("true", y_true)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite("test-RUC-Net/Big-junction-true.png", y_true)

# junction_pred = tuple(map(tuple, junction_pred))
# junction_true = tuple(map(tuple, junction_true))

# for junction in junction_pred:
#     print(f"{junction} is true positive: {junction in junction_true}")

# for i in range(len(junction_pred)):
#     print(np.linalg.norm(junction_pred[i] - junction_true[i]))
    
    
# combine_pred_true(y_true, y_pred, visualization=True, save_path="test-RUC-Net/combined.png")
