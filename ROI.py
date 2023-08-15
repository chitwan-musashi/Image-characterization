import cv2
import numpy as np

# Load the image
image_path = r"G:\AI Engineering\Co-ops\Chitwan Singh\Image Characterization\Defect Testing\14100\End Face\Co-axial\dent_co-axial_exposure_40000.jpg"
image = cv2.imread(image_path,  cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding
threshold_value = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
_, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Find contours of bright regions
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image and get bounding rectangles
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
bounding_rectangles = []
min_area_threshold = 100000  # Set the minimum area threshold here

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area_threshold:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        bounding_rectangles.append((x, y, w, h))

# Display the result
cv2.imshow('Bright Parts Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the bounding rectangles
for rect in bounding_rectangles:
    print(f"Bounding Rectangle: x={rect[0]}, y={rect[1]}, width={rect[2]}, height={rect[3]}")