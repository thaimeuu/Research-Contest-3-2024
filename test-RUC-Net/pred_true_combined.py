import numpy as np
import cv2


y_pred = cv2.imread("test-RUC-Net/y_pred_255_1C8-12-3.png", cv2.IMREAD_GRAYSCALE)
y_true = cv2.imread("test-RUC-Net/y_true_255_1C8-12-3.png", cv2.IMREAD_GRAYSCALE)

pred_white_px = np.argwhere(y_pred > 0)
true_white_px = np.argwhere(y_true > 0)

img_out = np.zeros(y_pred.shape + (3,))

for x, y in pred_white_px:
    img_out[x, y] = (0, 0, 255)
    
for x, y in true_white_px:
    if np.array_equal(img_out[x, y], np.array([0, 0, 0])):
        img_out[x, y] = (255, 0, 0)
    else:
        img_out[x, y] = (0, 255, 0)
    
# cv2.imshow("y_pred", y_pred)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("y_true", y_true)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imshow("pred_true_combined", img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("test-RUC-Net/pred_true_combined.png", img_out)
