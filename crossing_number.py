import os
import cv2
import numpy as np


"""
This script applies crossing number algorithm to detect rice panicle junctions

Input: None

Output: None (Save clustered image to save_path, save f1 image for evaluation)

How to use: Simply run the script
"""


def crossing_number() -> None:
    folder_path = r"skeleton_dataset"
    file_names = os.listdir(folder_path)

    for file_name in file_names:
        file_path = f"{folder_path}/{file_name}"
        img_in = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Uncomment to see original image
        # cv2.imshow(f"{file_name[:-4]}", img_in)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Algorithm treats white px intensity as 1
        img_in[img_in == 255] = 1
        white_px = np.argwhere(img_in > 0)
        centers: list[list[int]] = []

        # Crossing number
        for row, col in white_px:
            row, col = int(row), int(col)

            try:
                P1 = img_in[row, col + 1].astype("i")
                P2 = img_in[row - 1, col + 1].astype("i")
                P3 = img_in[row - 1, col].astype("i")
                P4 = img_in[row - 1, col - 1].astype("i")
                P5 = img_in[row, col - 1].astype("i")
                P6 = img_in[row + 1, col - 1].astype("i")
                P7 = img_in[row + 1, col].astype("i")
                P8 = img_in[row + 1, col + 1].astype("i")
            except:
                continue

            crossing_number = (
                abs(P2 - P1)
                + abs(P3 - P2)
                + abs(P4 - P3)
                + abs(P5 - P4)
                + abs(P6 - P5)
                + abs(P7 - P6)
                + abs(P8 - P7)
                + abs(P1 - P8)
            )

            crossing_number //= 2

            if crossing_number == 3 or crossing_number == 4:
                centers.append([row, col])

        # Convert white intensity back to 255
        img_in[img_in == 1] = 255
        img_out = cv2.cvtColor(img_in, cv2.COLOR_GRAY2RGB)
        for i in range(len(centers)):
            cv2.circle(img_out, (centers[i][1], centers[i][0]), 1, (0, 0, 255), -1)

        # Show result (Uncomment)
        # cv2.imshow(f"{file_name[:-4]}-crossing_number", img_out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Save file
        save_path = rf"clustered_skeleton/Crossing-number/gradient-based-optimization/{file_name}"
        cv2.imwrite(save_path, img_out)

        print(f"Successfully generated {file_name}")

        # Create labels for f1_score evaluation
        img_f1 = np.zeros([256, 256]).astype(np.uint8)
        for row, col in centers:
            img_f1[row, col] = 1

        f1_save_path = rf"F1_score/crossing-number/Zhang-Suen/{file_name}"
        cv2.imwrite(f1_save_path, img_f1)
        print(f"Successfully generated f1 image: {file_name}")


if __name__ == "__main__":
    crossing_number()
