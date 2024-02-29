import cv2
import numpy as np
from sklearn.metrics import f1_score


file_path = r"clustered_skeleton/Crossing-number/f1_score/1C8-10-1.png"
img_in = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

values, freq = np.unique(img_in, return_counts=True)
print(values, freq)

# img_in = cv2.resize(img_in, [2658, 1773])

# cv2.imshow("image", img_in)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite('test.png', img_in)


