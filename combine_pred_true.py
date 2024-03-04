import numpy as np
import cv2


"""
This script is an Utility
"""

def combine_pred_true(y_true: np.array, y_pred: np.array, visualization=False, save_path=None) -> np.array:
    """
    Combine predicted junctions and true junctions on the same image
    predicted junctions are marked in red
    true junctions are marked in blue
    true positives are marked in green

    Args:
        y_true (np.array): grayscale target (junctions are in white on black background)
        y_pred (np.array): grayscale prediction (junctions are in white on black background)
        visualization (bool, optional): _description_. Defaults to False.
        save_path: Defaults to None

    Returns:
        np.array: combined ndarray
    """
    true_white_px = np.argwhere(y_true > 0)
    pred_white_px = np.argwhere(y_pred > 0)
    
    img_out = np.zeros(y_pred.shape + (3,)).astype(np.uint8)
    
    for x, y in true_white_px:
        img_out[x, y] = (255, 0, 0)  # blue
        
    for x, y in pred_white_px:
        if np.array_equal(img_out[x, y], np.array([0, 0, 0])):
            img_out[x, y] = (0, 0, 255)  # red
        else:
            img_out[x, y] = (0, 255, 0)  # green
            
    if visualization:
        cv2.imshow("Combined image", img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    if save_path:
        cv2.imwrite(save_path, img_out)
    
    return img_out

if __name__ == "__main__":
    y_true = cv2.imread("test1.png", cv2.IMREAD_GRAYSCALE)
    y_pred = cv2.imread("test2.png", cv2.IMREAD_GRAYSCALE)
    combine_pred_true(y_true, y_pred, visualization=True, save_path="D:/HRG/SVNCKH 3-2024/test3.png")
    