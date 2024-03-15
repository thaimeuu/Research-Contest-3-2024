import os
from vertices_coordinates import find_vertices
import cv2

"""
Inputs:
    Grayscale image path of Panicle
    XML Result path by Jouannic Paper 
    
Outputs:
    Original Image with Vertices added with color based on their types
    Save this new image to folder labelled_dataset/ 
    
Instructions:
    Run rgb_2_gray.py first then take its output as this file's input

"""

def main():
    folder_path = "dataset_1_grayscale"
    # folder_path = "dataset_2_grayscale/.Asian-African panel_CIAT/Asian-African panel_New"
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        # Read grayscale img
        path = folder_path + "/" + file_name
        img = cv2.imread(path)

        xml_file_path = f"dataset_1_xml/{file_name[:-4]}.ricepr"
        # xml_file_path = f"dataset_2_xml/.Asian-African panel_CIAT/Asian-African panel_New/{file_name[:-4]}.ricepr"
        print(f"===\nExtracting junctions from {xml_file_path}\n===")

        # lists of junctions' coordinate
        generating = find_vertices("Generating", xml_file_path)
        primary = find_vertices("Primary", xml_file_path)
        secondary = find_vertices("Seconday", xml_file_path)  # not typo
        tertiary = find_vertices("Tertiary", xml_file_path)
        quaternary = find_vertices("Quaternary", xml_file_path)
        # terminal = find_vertices("End", xml_file_path)

        # Add junctions to grayscale img
        for a, b in generating:
            cv2.circle(img, (a, b), 15, (0, 255, 255), cv2.FILLED)
        for a, b in primary:
            cv2.circle(img, (a, b), 15, (255, 255, 255), cv2.FILLED)
        for a, b in secondary:
            cv2.circle(img, (a, b), 15, (255, 0, 0), cv2.FILLED)
        # for a, b in tertiary:
        #     cv2.circle(img, (a, b), 8, (0, 255, 0), cv2.FILLED)
        # for a, b in quaternary:
        #     cv2.circle(img, (a, b), 8, (255, 255, 0), cv2.FILLED)
        # for a, b in terminal:
        #     cv2.circle(img, (a, b), 8, (0, 0, 255), cv2.FILLED)

        # Save file
        save_path = rf"dataset_1_labelled/{file_name}"
        # save_path = rf"dataset_2_labelled/.Asian-African panel_CIAT/Asian-African panel_New/{file_name}"
        cv2.imwrite(save_path, img)
        
        print(f"Successfully created {save_path}\n")
        
        # break


if __name__ == "__main__":
    main()
