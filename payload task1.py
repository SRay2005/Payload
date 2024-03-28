import cv2
import numpy as np

# Read the image
img = cv2.imread('clock_tower.jpg')

# Define binning factor
binning_factor = 4

# Perform pixel binning
binned_img = cv2.resize(img, (img.shape[1] // binning_factor, img.shape[0] // binning_factor))

# Save the binned image
cv2.imwrite('clock_tower_binned.jpg', binned_img)

# Display the original and binned images
cv2.imshow('Original Image', img)
cv2.imshow('Binned Image', binned_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
