from skimage.morphology import skeletonize
import numpy as np
import cv2

"""
Inputs:
    "Rice Panicle" binary image

Outputs:
    Skeleton image of "Rice Panicle"
    saves the skeleton image under the same name as the initial path and places it in skeleton_dataset/
"""


def main():
    panicles = ["image_cornflower/1C8-6-1.png",
                "image_cornflower/1C8-6-2.png",
                "image_cornflower/1C8-6-3.png",
                "image_cornflower/1C8-7-1.png",
                "image_cornflower/1C8-7-2.png",
                "image_cornflower/1C8-7-3.png",
                "image_cornflower/1C8-8-1.png",
                "image_cornflower/1C8-8-2.png",
                "image_cornflower/1C8-8-3.png",
                "image_cornflower/1C8-9-1.png",
                "image_cornflower/1C8-9-2.png",
                "image_cornflower/1C8-9-3.png",
                "image_cornflower/1C8-10-1.png",
                "image_cornflower/1C8-10-2.png",
                "image_cornflower/1C8-10-3.png",
                "image_cornflower/1C8-11-1.png",
                "image_cornflower/1C8-11-2.png",
                "image_cornflower/1C8-11-3.png",
                "image_cornflower/1C8-12-1.png",
                "image_cornflower/1C8-12-2.png",
                "image_cornflower/1C8-12-3.png"]
    
    for path in panicles:
    
        # Read binary image
        # binary_img_path = "test_binary_img.png"
        binary_img_path = path
        img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)

        # Extract Skeleton
        skeleton = skeletonize(img, method="zhang").astype(np.uint8) * 255

        # Save file
        # cv2.imwrite(f"skeleton_dataset/{binary_img_path}", skeleton)
        cv2.imwrite(f"skeleton_dataset/{binary_img_path[17:]}", skeleton)


if __name__ == "__main__":
    main()
