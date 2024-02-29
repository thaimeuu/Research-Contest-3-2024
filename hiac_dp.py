import os
import sys

sys.path.append("code availability/HIAC-main")

from HIAC import *
import numpy as np
import cv2
import matplotlib.pyplot as plt


"""
This script applies DPC on a HIAC-ameliorated-image to find rice panicle's junctions

HIAC parameters are set as follows:
- k: 30
- T: 0.7
- d: 5
- threshold:  # record on parameter-hiac.xlsx file

DPC parameters are set as follows:
- k = 25 (number of clusters)
- percent = 2
- kernel = cutoff

How to use: manually specifying file_name, choose parameters and run the script
"""


def hiac_dp() -> None:
    folder_path = r"skeleton_dataset"
    file_names = os.listdir(folder_path)
    
    file_name = file_names[20]

    # Specify HIAC parameters
    k, T, d, threshold = (30, 0.7, 5, 0.9925)

    file_path = f"{folder_path}/{file_name}"
    img_in = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    print(f"image shape: {img_in.shape}")

    white_px = np.argwhere(img_in > 0)

    # Plot original white px (Uncomment)
    # plt.scatter(white_px[:, 1], white_px[:, 0], s=1)
    # plt.gca().invert_yaxis()
    # plt.title("Original white_px")
    # plt.show()

    # Normalize white_px
    white_px_norm = white_px.copy().astype(np.float64)
    for col in range(white_px_norm.shape[1]):
        max_ = max(white_px_norm[:, col])
        min_ = min(white_px_norm[:, col])
        if max_ == min_:
            continue
        for row in range(white_px_norm.shape[0]):
            white_px_norm[row, col] = (white_px_norm[row, col] - min_) / (max_ - min_)

    norm_data_save_path = "clustered_skeleton/HIAC-DPC/HIAC-info"
    np.savetxt(os.path.join(norm_data_save_path, file_name[:-4] + "_normalized.txt"), white_px_norm)
    
    # Plot normalized white px (Uncomment)
    # plt.scatter(white_px_norm[:, 1], white_px_norm[:, 0], s=1)
    # plt.gca().invert_yaxis()
    # plt.title("Normalized white_px")
    # plt.show()

    # Call HIAC (important)
    photoPath = rf"clustered_skeleton/HIAC-DPC/HIAC-info/{file_name[:-4]}_decision_graph.png"
    distanceTGP = TGP(white_px_norm, k, photoPath, threshold)
    # for object i, if j is invalid-neighbor of i, neighbor_index[i][j] = -1
    neighbor_index = prune(white_px_norm, k, threshold, distanceTGP)

    # ameliorate the dataset by d time-segments
    for i in range(d):  # ameliorated the dataset by d time-segments
        bata = shrink(white_px_norm, k, T, neighbor_index)
        ameliorated_white_px = bata

    ameliorated_data_save_path = "clustered_skeleton/HIAC-DPC/HIAC-info"
    np.savetxt(os.path.join(ameliorated_data_save_path, file_name[:-4] + "_ameliorated_by_HIAC.txt"), ameliorated_white_px)
    
    # Call DPC (important) -> get the labels
    cl: list[int] = DPC(ameliorated_white_px, k=25, ratio=2, kernel="cutoff", decision_graph=False)

    # Plot clustered ameliorated white px (Uncomment)
    plt.scatter(ameliorated_white_px[:, 1], ameliorated_white_px[:, 0], c=cl, s=1)
    plt.gca().invert_yaxis()
    plt.title("Ameliorated white_px")
    plt.savefig(ameliorated_data_save_path + rf"/{file_name[:-4]}_ameliorated.png", dpi=300)
    plt.show()

    # Get clusters centers coordinate to plot on the original image
    clusters, freq = np.unique(cl, return_counts=True)
    centers = np.zeros([len(clusters), 2])
    for i, cluster in enumerate(clusters):
        centers[i, :] = np.mean(white_px[cl == cluster], axis=0)

    centers = centers.astype("i")

    # Draw centers on original image
    img_out = cv2.cvtColor(img_in, cv2.COLOR_GRAY2RGB)
    for i in range(len(centers)):
        cv2.circle(img_out, (centers[i, 1], centers[i, 0]), 1, (0, 0, 255), -1)

    # Uncomment 3 lines below to view the clustered image
    # cv2.imshow("DPC on HIAC-ameliorated-white px", img_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save image
    save_path = rf"clustered_skeleton/HIAC-DPC/{file_name}"
    cv2.imwrite(save_path, img_out)
    
    print(f"Successfully generated {file_name}")
        
        
if __name__ == "__main__":
    hiac_dp()
