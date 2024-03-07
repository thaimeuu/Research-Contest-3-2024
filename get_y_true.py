import os
import numpy as np
import cv2

from new_coordinate import new_coordinate
from vertices_coordinates import find_vertices

def get_y_true() -> None:
    """
    
    """
    xml_folder = "dataset_2_xml/.Asian-African panel_CIAT/Asian-African panel_New"
    xml_file_names = os.listdir(xml_folder)
    
    for xml_file in xml_file_names:
        xml_file_path = xml_folder + "/" + xml_file
        save_name = xml_file[:-7]  # 1C8-7-1
        print(f"\nExtracting {xml_file} ...........")
        
        generating = find_vertices("Generating", xml_file_path)
        primary = find_vertices("Primary", xml_file_path)
        secondary = find_vertices("Seconday", xml_file_path)
        tertiary = find_vertices("Tertiary", xml_file_path)
        
        y_true_1 = np.zeros([256, 256]).astype(np.uint8)
        y_true_25 = np.zeros([256, 256]).astype(np.uint8)
        
        original_img = cv2.imread(f"dataset_2/.Asian-African panel_CIAT/Asian-African panel_CIAT/Asian-African panel_New/riceproj/images/{save_name}.JPG", cv2.IMREAD_GRAYSCALE)
        old_shape = (original_img.shape[1], original_img.shape[0])
        
        for a, b in generating:
            new_a, new_b = new_coordinate([a, b], old_shape, [256, 256])
            y_true_1[new_b, new_a] = 255
            # cv2.circle(y_true_25, (new_a, new_b), 2, 255, -1)  # each junction takes up 25 pixels
            y_true_25[new_b -2 : new_b + 3, new_a - 2 : new_a + 3] = 255  # each junction takes up 25 pixels
            
        for a, b in primary:
            new_a, new_b = new_coordinate([a, b], old_shape, [256, 256])
            y_true_1[new_b, new_a] = 255
            # cv2.circle(y_true_25, (new_a, new_b), 2, 255, -1)
            y_true_25[new_b -2 : new_b + 3, new_a - 2 : new_a + 3] = 255
            
        for a, b in secondary:
            new_a, new_b = new_coordinate([a, b], old_shape, [256, 256])
            y_true_1[new_b, new_a] = 255
            # cv2.circle(y_true_25, (new_a, new_b), 2, 255, -1)
            y_true_25[new_b -2 : new_b + 3, new_a - 2 : new_a + 3] = 255
            
        for a, b in tertiary:
            new_a, new_b = new_coordinate([a, b], old_shape, [256, 256])
            y_true_1[new_b, new_a] = 255
            # cv2.circle(y_true_25, (new_a, new_b), 2, 255, -1)
            y_true_25[new_b -2 : new_b + 3, new_a - 2 : new_a + 3] = 255
            
        cv2.imwrite(f"dataset_2_y_true/.Asian-African panel_CIAT/Asian-African panel_New/crossing_number/y_true_1/{save_name}.png", y_true_1)
        cv2.imwrite(f"dataset_2_y_true/.Asian-African panel_CIAT/Asian-African panel_New/crossing_number/y_true_25/{save_name}.png", y_true_25)
        
        break
    
    
if __name__ == "__main__":
    get_y_true()
    