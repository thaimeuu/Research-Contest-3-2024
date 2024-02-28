import os
import sys

sys.path.append("code availability/HIAC-main")

from DPC import DPC
import cv2
import numpy as np


"""
This script applies DPC (density peaks clustering to find rice panicle's junctions)

DPC parameters are set as follows:
- k = 25 (number of clusters)
- percent = 0.75
- kernel = cutoff

How to use: Just run the script
"""


def dp() -> None:
    """
    density peaks clustering

    arguments: none

    returns: none (save image)
    """
    folder_path = r"skeleton_dataset"
    file_names = os.listdir(folder_path)

    for file_name in file_names:

        file_path = f"{folder_path}/{file_name}"

        img_in = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        print(f"image shape: {img_in.shape}")

        # Uncomment 3 lines below to view the original image
        # cv2.imshow("original", img_in)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        white_px = np.argwhere(img_in > 0)
        print(f"white pixels shape: {white_px.shape}")
        print(f"First 5 elements:\n{white_px[:5, :]}")

        # Call DPC
        labels = DPC(white_px, 25, ratio=0.75, kernel="cutoff", decision_graph=False)
        print(f"Labels shape: {labels.shape}")
        clusters, freq = np.unique(labels, return_counts=True)
        print(f"labels unique values and their frequencies: {clusters}, {freq}")

        # Plot cluster centers
        centers = np.zeros([len(clusters), 2])
        for i, cluster in enumerate(clusters):
            centers[i, :] = np.mean(white_px[labels == cluster], axis=0)

        centers = centers.astype("i")
        print(centers)

        img_out = cv2.cvtColor(img_in, cv2.COLOR_GRAY2RGB)
        for i in range(len(centers)):
            cv2.circle(img_out, (centers[i, 1], centers[i, 0]), 1, (0, 0, 255), -1)

        # Uncomment 3 lines below to view the clustered image
        # cv2.imshow("DPC", img_out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Save image
        save_path = rf"clustered_skeleton/DPC/{file_name}"
        cv2.imwrite(save_path, img_out)


if __name__ == "__main__":
    dp()
