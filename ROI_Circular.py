import cv2
import numpy as np

# Load the image
image_path = r"g:\AI Engineering\Co-ops\Chitwan Singh\Plane Distortion\Endface\End_face_complete_part1_91.5_degree.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding
threshold_value = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
_, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Find contours of bright regions
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert the image to color (BGR) format using cv2.merge
output_image = cv2.imread(image_path)

# Define the minimum area threshold
min_area_threshold = 10000

# Draw contours on the original image and get bounding circles
bounding_circles = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area_threshold:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(output_image, center, radius, (0, 0, 255), 2)
        bounding_circles.append((center, radius))

# Display the result
cv2.imshow('Bright Parts Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("image.jpg", output_image)

# Print the bounding circles
for circle in bounding_circles:
    center = circle[0]
    radius = circle[1]
    print(f"Bounding Circle: center={center}, radius={radius}")
