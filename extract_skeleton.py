import cv2
import numpy as np
from skimage.morphology import thin
from scipy.ndimage import distance_transform_edt


def get_2d_skeleton_image(mask):
    # distance transform and thinning
    dist = distance_transform_edt(mask)
    thinned = thin((mask > 0).astype(np.uint8)).astype(np.float32)
    y, x = np.where(thinned > 0)
    indices = np.argsort(dist[y, x])[::-1]
    x, y = x[indices], y[indices]

    # create a blank image to draw the skeleton
    skeleton_image = np.zeros_like(mask, dtype=np.uint8)

    # mark skeleton points on the image
    for idx in range(x.shape[0]):
        skeleton_image[y[idx], x[idx]] = 255

    return skeleton_image


# Example usage:
binary_image = cv2.imread("test_img.png", cv2.IMREAD_GRAYSCALE)
skeleton_image = get_2d_skeleton_image(binary_image)

# Visualize or further process the skeleton image as needed
cv2.imshow("Skeleton Image", skeleton_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
