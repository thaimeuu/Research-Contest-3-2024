from skimage.morphology import skeletonize
import numpy as np
import cv2

def main():
    binary_img_path = "test_binary_img.png"
    img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
    
    skeleton = skeletonize(img, method="zhang").astype(np.uint8) * 255
    cv2.imwrite(f'skeleton_dataset/{binary_img_path}', skeleton)
    

if __name__ == "__main__":
    main()
    