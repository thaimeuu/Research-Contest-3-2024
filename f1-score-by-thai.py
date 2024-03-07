import os
import numpy as np
import cv2
from combine_pred_true import combine_pred_true

"""
Method:
Compare 2 images:
    - y_pred with each predicted junction taking up 1 pixel
    - y_true_25 with each true junction taking up 25 pixels (central pixel being true junction -> y_true_1)
Evaluate: 
    - If a predicted junction is the same as 1 out of 25 true-junction pixels, count as true positives 
"""

def f1_by_thai(y_true_25: np.array, y_true_1: np.array, y_pred: np.array, return_precision=True, return_recall=True) -> float:
    """
    ultimate f1-score by thai ^_^

    Args:
        y_true_25 (np.array): any size allowed (Each junction takes up 25 pixels (i.e., cv2.circle(radius=2)) (important))
        y_true_1 (np.array): any size allowed (Each junction takes up 1 pixels (important)
        y_pred (np.array): any size allowed (Each junction takes up 1 pixel)
        return_precision (bool): Defaults:True
        return_recall (bool): Defaults:True

    Returns:
        f1: float: f1 score
    """
    true_25, true_1, pred = np.copy(y_true_25), np.copy(y_true_1), np.copy(y_pred)

    junction_true_25 = np.argwhere(true_25 > 0)
    junction_true_1 = np.argwhere(true_1 > 0)
    junction_pred = np.argwhere(pred > 0)

    true_25 = tuple(map(tuple, junction_true_25))
    n_true_clusters = len(junction_true_1)
    n_pred_clusters = len(junction_pred)
    pred = tuple(map(tuple, junction_pred))

    true_pos, false_pos, false_neg = (0, 0, 0)

    for junction in pred:
        if junction in true_25:
            true_pos += 1
  
    false_pos = n_pred_clusters - true_pos
    false_neg = n_true_clusters - true_pos
    
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score if not (return_precision and return_recall) else [f1_score, precision, recall] if (return_precision and return_recall) else [f1_score, precision] if (return_precision and not return_recall) else [f1_score, recall]


if __name__ == "__main__":
    folder_name = "y_pred/crossing-number/Zhang-Suen/RUC-Net"
    file_names = os.listdir(folder_name)
    f1_record = []
    for file_name in file_names:
        print(f"\n{file_name}")
        y_true_25 = cv2.imread(f"y_true/crossing_number/y_true_25/{file_name}", cv2.IMREAD_GRAYSCALE)
        y_true_1 = cv2.imread(f"y_true/crossing_number/y_true_1/{file_name}", cv2.IMREAD_GRAYSCALE)
        # y_pred = cv2.imread(f"y_pred/crossing-number/gradient-based-optimization/RUC-Net/{file_name}", cv2.IMREAD_GRAYSCALE)
        y_pred = cv2.imread(f"y_pred/crossing-number/Zhang-Suen/RUC-Net/{file_name}", cv2.IMREAD_GRAYSCALE)
        f1_score, precision, recall = f1_by_thai(y_true_25, y_true_1, y_pred)
        f1_record.append(f1_score)
        print(f1_score, precision, recall)
        
        # file_path = "F1-record.txt"
        # with open(file_path, 'a') as f:
        #     f.write(f"{file_name}: F1 = {f1_score}\n")
        
        # combine_pred_true(y_true_25, y_pred, visualization=True)
    
    print(min(f1_record), max(f1_record), np.mean(f1_record))