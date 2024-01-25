from skimage.morphology import skeletonize
import numpy as np
import cv2

"""
This script extracts the skeleton image of/from a "Rice Panicle" binary image

It also saves the skeleton image under the same name as the initial path and places it in skeleton_dataset/
"""


def main():
    # Read binary image
    binary_img_path = "test_binary_img.png"
    img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)

    # Extract Skeleton
    skeleton = skeletonize(img, method="zhang").astype(np.uint8) * 255
    
    # Save file
    cv2.imwrite(f"skeleton_dataset/{binary_img_path}", skeleton)


if __name__ == "__main__":
    main()
