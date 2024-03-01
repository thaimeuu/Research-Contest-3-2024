import os
import cv2
import numpy as np
from sklearn.metrics import f1_score
from new_coordinate import new_coordinate
from vertices_coordinates import find_vertices


def main() -> None:

    f1_folder_path = "F1_score/crossing-number/Zhang-Suen"
    f1_file_names = os.listdir(f1_folder_path)

    for f1_file_name in f1_file_names:
        f1_file_path = f1_folder_path + "/" + f1_file_name

        # y_pred used for f1_score
        img_pred = cv2.imread(f1_file_path, cv2.IMREAD_GRAYSCALE)
        new_shape = img_pred.shape  # Used for new_coordinate()

        # Get junctions coordinate (ground truth)
        xml_folder_path = "Jouannic_xml"
        xml_file_path = xml_folder_path + "/" + f1_file_name[:-4] + ".ricepr"
        print(f"=====\nExtracting vertices coordinate from {xml_file_path}...\n=====")

        generating = find_vertices("Generating", xml_file_path)
        primary = find_vertices("Primary", xml_file_path)
        secondary = find_vertices("Seconday", xml_file_path)  # not typo
        tertiary = find_vertices("Tertiary", xml_file_path)
        quaternary = find_vertices("Quaternary", xml_file_path)

        # Initialize y_true used for f1_score
        y_true = np.zeros(img_pred.shape).astype(np.uint8)

        # New coordinates and add them to y_true
        old_shape = (2658, 1773)

        for old_coordinates in generating:
            new_x, new_y = new_coordinate(old_coordinates, old_shape, new_shape)
            y_true[new_y, new_x] = 1
        for old_coordinates in primary:
            new_x, new_y = new_coordinate(old_coordinates, old_shape, new_shape)
            y_true[new_y, new_x] = 1
        for old_coordinates in secondary:
            new_x, new_y = new_coordinate(old_coordinates, old_shape, new_shape)
            y_true[new_y, new_x] = 1
        for old_coordinates in tertiary:
            new_x, new_y = new_coordinate(old_coordinates, old_shape, new_shape)
            y_true[new_y, new_x] = 1
        for old_coordinates in quaternary:
            new_x, new_y = new_coordinate(old_coordinates, old_shape, new_shape)
            y_true[new_y, new_x] = 1

        # Check ground truth vertices (Uncomment immediately after use)
        # y_true[y_true == 1] = 255
        # cv2.imshow("ground truth", y_true)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Check ground truth vertices (Uncomment immediately after use)
        # img_pred[img_pred == 1] = 255
        # cv2.imshow("prediction", img_pred)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        y_true = y_true.flatten()
        y_pred = img_pred.flatten()

        print(f"y_true: {y_true.shape}, unique: {np.unique(y_true, return_counts=True)}")
        print(f"y_pred: {y_pred.shape}, unique: {np.unique(y_pred, return_counts=True)}")

        F1 = f1_score(y_true, y_pred, average="binary")

        print(f"Successfully calculated F1 score for {f1_file_name}: {F1}\n")


if __name__ == "__main__":
    main()
