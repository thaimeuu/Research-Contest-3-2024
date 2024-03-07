import os
import cv2

"""
Inputs:
    RGB image path of a panicle
    
Outputs:
    Grayscale image saved in folder grayscale_dataset/
    
Instructions:
    Run this file first and take the output as input for label_dataset.py
""" 


def rgb_2_gray(rgb_path):
    """
    rgb 2 grayscale

    Args:
        rgb_path (str): path
        
    Returns:
        grayscale img
    """
    file_names = os.listdir(rgb_path)
    file_names.remove('.DS_Store')
    file_names.remove('SinZoom.jpg')
    for file in file_names:
        file_path = rgb_path + "/" + file
        
        rgb = cv2.imread(file_path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        save_path = f"dataset_2_grayscale/.Asian-African panel_CIAT/Asian-African panel_New/{file}"
        cv2.imwrite(save_path,gray)
        
        break

    
if __name__ == "__main__":
    rgb_path = "dataset_2/.Asian-African panel_CIAT/Asian-African panel_CIAT/Asian-African panel_New/riceproj/images"
    rgb_2_gray(rgb_path)
    