import os
import sys

sys.path.append("code availability/HIAC-main")

from HIAC import *
import numpy as np
import cv2
import matplotlib.pyplot as plt


def hiac_dp():
    folder_path = r"skeleton_dataset"
    file_names = os.listdir(folder_path)
    
    for file_name in file_names:
        # Specify HIAC parameters
        k, T, d, threshold = (6, 0.3, 4, 1.514)

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
        
        
        
        break
        

if __name__ == "__main__":
    hiac_dp()
