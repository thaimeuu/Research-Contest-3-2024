import os
import cv2
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from vertices_coordinates import find_vertices
from combine_pred_true import combine_pred_true


def F1_alternative(return_confusion_matrix=False, return_counts=False, combine_results=False) -> None:

    # Toggle between Zhang-Suen and gradient-based-optimization
    # f1_folder_path = "F1_score/crossing-number/Zhang-Suen"
    f1_folder_path = "F1_score/crossing-number/gradient-based-optimization"
    f1_file_names = os.listdir(f1_folder_path)

    for f1_file_name in f1_file_names:
        print(f"====================\nEXAMINING {f1_file_name} from {f1_folder_path}...")
        f1_file_path = f1_folder_path + "/" + f1_file_name

        # y_pred used for f1_score
        y_pred = cv2.imread(f1_file_path, cv2.IMREAD_GRAYSCALE)
        y_pred_copy = np.copy(y_pred)

        # Get junctions coordinate (ground truth)
        xml_folder_path = "Jouannic_xml"
        xml_file_path = xml_folder_path + "/" + f1_file_name[:-4] + ".ricepr"
        print(f"=====\nExtracting vertices coordinate from {xml_file_path}...\n=====")

        generating = find_vertices("Generating", xml_file_path)
        primary = find_vertices("Primary", xml_file_path)
        secondary = find_vertices("Seconday", xml_file_path)  # not typo
        tertiary = find_vertices("Tertiary", xml_file_path)
        quaternary = find_vertices("Quaternary", xml_file_path)

        # New coordinates and add them to y_true
        original_image_path = f"Jouannic_dataset/{f1_file_name[:-4]}.JPG"
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Vary marker_size range and capture best F1-score
        f1_record = []
        cm_record = []
        
        for marker_size in range(10, 20):    
            # Initialize y_true used for f1_score
            y_true = np.zeros(original_image.shape).astype(np.uint8)
            
            for x, y in generating:
                cv2.circle(y_true, [x, y], marker_size, 255, -1)
            for x, y in primary:
                cv2.circle(y_true, [x, y], marker_size, 255, -1)
            for x, y in secondary:
                cv2.circle(y_true, [x, y], marker_size, 255, -1)
            for x, y in tertiary:
                cv2.circle(y_true, [x, y], marker_size, 255, -1)
            for x, y in quaternary:
                cv2.circle(y_true, [x, y], marker_size, 255, -1)
            
            
            y_true = cv2.resize(y_true, [256, 256], interpolation=cv2.INTER_NEAREST)
            
            y_true[y_true == 255] = 1
            y_pred[y_pred == 255] = 1

            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            
            cm = confusion_matrix(y_true, y_pred)
            cm_record.append(cm)

            F1 = f1_score(y_true, y_pred, average="binary")
            f1_record.append(F1)

        print(f"Successfully calculated F1 score for {f1_file_name}: {max(f1_record)} with marker_size = {np.argmax(f1_record) + 10}\n====================")
        
        if return_confusion_matrix:
            print(f"---\nConfusion matrix:\n{cm_record[np.argmax(f1_record)]}\n---")
        
        if return_counts:
            y_true = np.zeros(original_image.shape).astype(np.uint8)
            
            for x, y in generating:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            for x, y in primary:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            for x, y in secondary:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            for x, y in tertiary:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            for x, y in quaternary:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            
            y_true = cv2.resize(y_true, [256, 256], interpolation=cv2.INTER_NEAREST)
            
            print("Return counts:")
            print(f"y_true: {np.unique(y_true, return_counts=True)}")
            print(f"y_pred: {np.unique(y_pred_copy, return_counts=True)}")
            
        if combine_results:
            y_true = np.zeros(original_image.shape).astype(np.uint8)
            
            for x, y in generating:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            for x, y in primary:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            for x, y in secondary:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            for x, y in tertiary:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            for x, y in quaternary:
                cv2.circle(y_true, [x, y], np.argmax(f1_record) + 10, 255, -1)
            
            y_true = cv2.resize(y_true, [256, 256], interpolation=cv2.INTER_NEAREST)
            
            combine_pred_true(y_true, y_pred_copy, visualization=True)
            
        print("")
        
            


if __name__ == "__main__":
    F1_alternative(return_confusion_matrix=True, return_counts=True, combine_results=False)
