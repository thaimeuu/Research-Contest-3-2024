import cv2
import numpy as np

def rgb_to_gray(rgb_image):
    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return gray_image


input_image_path = 'test_RGB_img.JPG'
rgb_image = cv2.imread(input_image_path)


# Convert RGB to grayscale
grayscale_image = rgb_to_gray(rgb_image)

# Display the original and grayscale images
# cv2.imshow('Original RGB Image', rgb_image)
# cv2.imshow('Grayscale Image', grayscale_image)

# Save the grayscale image to a file
cv2.imwrite(r'grayscale_dataset/grayscale_output.jpg', grayscale_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
