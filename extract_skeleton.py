import os
from skimage.morphology import skeletonize
import numpy as np
import cv2

"""
Inputs:
    "Rice Panicle" binary image

Outputs:
    Skeleton image of "Rice Panicle"
    saves the skeleton image under the same name as the initial path and places it in skeleton_dataset/
"""


def main():
    folder_name = "RUC-Net images"
    file_names = os.listdir(folder_name)
    for file_name in file_names:
        path = folder_name + "/" + file_name
        
        # Read binary image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Extract Skeleton
        skeleton = skeletonize(img, method="zhang").astype(np.uint8) * 255

        # Save file
        # cv2.imwrite(f"skeleton_dataset/{binary_img_path}", skeleton)
        cv2.imwrite(f"skeleton_dataset/RUC-Net/{file_name}", skeleton)
        
        break


if __name__ == "__main__":
    main()
