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
    rgb = cv2.imread(rgb_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    
    return gray

def main():
    rgb_path = r"Jouannic_dataset/1C8-12-3.JPG"
    gray = rgb_2_gray(rgb_path)
    cv2.imwrite(f"grayscale_dataset/{rgb_path[17:]}", gray)
    
if __name__ == "__main__":
    main()
    