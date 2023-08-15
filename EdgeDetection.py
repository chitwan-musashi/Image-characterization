import cv2
import numpy as np

def edge_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Calculate the dimensions of the image
    image_height, image_width, _ = image.shape

    image = cv2.resize(image, (image_width // 4, image_height // 4))


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calculate gradient using Sobel operator
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude and direction of gradients
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Perform non-maximum suppression
    suppressed = np.zeros_like(gradient_magnitude)
    angle_quantized = np.round(gradient_direction * (5.0 / np.pi)) % 4
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            if angle_quantized[i, j] == 0 and gradient_magnitude[i, j] == np.max([gradient_magnitude[i, j], gradient_magnitude[i, j + 1], gradient_magnitude[i, j - 1]]):
                suppressed[i, j] = gradient_magnitude[i, j]
            elif angle_quantized[i, j] == 1 and gradient_magnitude[i, j] == np.max([gradient_magnitude[i, j], gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]):
                suppressed[i, j] = gradient_magnitude[i, j]
            elif angle_quantized[i, j] == 2 and gradient_magnitude[i, j] == np.max([gradient_magnitude[i, j], gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]):
                suppressed[i, j] = gradient_magnitude[i, j]
            elif angle_quantized[i, j] == 3 and gradient_magnitude[i, j] == np.max([gradient_magnitude[i, j], gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]):
                suppressed[i, j] = gradient_magnitude[i, j]

    # Define high and low thresholds for double thresholding
    high_threshold = 100
    low_threshold = 50

    # Perform double thresholding
    strong_edges = suppressed > high_threshold
    weak_edges = np.logical_and(suppressed >= low_threshold, suppressed <= high_threshold)

    # Perform edge tracking by hysteresis
    edges = np.zeros_like(image)
    edges[strong_edges] = 255
    while np.sum(weak_edges) > 0:
        weak_edge_indices = np.argwhere(weak_edges)
        current_weak_edge = weak_edge_indices[0]
        edges[current_weak_edge[0], current_weak_edge[1]] = 255
        weak_edges[current_weak_edge[0], current_weak_edge[1]] = False
        neighbors = np.argwhere(np.logical_and(weak_edges, np.abs(gradient_direction - gradient_direction[current_weak_edge[0], current_weak_edge[1]]) <= np.pi / 4))
        for neighbor in neighbors:
            edges[neighbor[0], neighbor[1]] = 255
            weak_edges[neighbor[0], neighbor[1]] = False
 
    return edges

    '''# Display the detected edges
    cv2.imshow("Original Image", image)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

# Specify the path to your image
image_path = "G:\AI Engineering\Co-ops\Chitwan Singh\Image Characterization\Defect Testing\\14100\Chip Defect\\result1\chip_J1_exposure_27500.jpg"

# Call the edge_detection function with the image path
edge_detection(image_path)
