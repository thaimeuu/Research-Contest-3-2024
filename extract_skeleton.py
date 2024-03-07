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


def extract_skeleton(dataset: int):
    if dataset == 1:
        folder_name = "RUC-Net images/dataset_1"
        file_names = os.listdir(folder_name)
        for file_name in file_names:
            path = folder_name + "/" + file_name
            
            # Read binary image
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # Extract Skeleton
            skeleton = skeletonize(img, method="zhang").astype(np.uint8) * 255

            # Save file
            # cv2.imwrite(f"skeleton_dataset/{binary_img_path}", skeleton)
            cv2.imwrite(f"dataset_1_skeleton_Zhang_Suen/RUC-Net/{file_name}", skeleton)
            
            break
        
    if dataset == 2:
        folder_name = "dataset_2_binary/.Asian-African panel_CIAT/Asian-African panel_New"
        file_names = os.listdir(folder_name)
        for file_name in file_names:
            path = folder_name + "/" + file_name
            
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            skeleton = skeletonize(img, method="zhang").astype(np.uint8) * 255
            
            save_path = f"dataset_2_skeleton_Zhang_Suen/.Asian-African panel_CIAT/Asian-African panel_New/{file_name}"
            cv2.imwrite(save_path, skeleton)
            
            break


if __name__ == "__main__":
    # extract_skeleton(1)
    extract_skeleton(2)
