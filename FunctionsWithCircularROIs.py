import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy

def calculate_circular_ROIs(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding
    threshold_value = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours of bright regions
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image and get bounding circles
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bounding_circles = []
    min_area_threshold = 10000  # Set the minimum area threshold here

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output_image, center, radius, (0, 0, 255), 2)
            bounding_circles.append((center, radius))

    return bounding_circles

def calculate_contrast(image_path, rois):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a blank mask
    mask = np.zeros_like(image)

    # Set pixel values inside ROIs to 255 (white)
    for roi in rois:
        center, radius = roi
        cv2.circle(mask, center, radius, 255, -1)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    # Calculate contrast of the masked image
    contrast = masked_image.std()

    return contrast

def calculate_sharpness(image_path):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    array = np.asarray(image, dtype=np.int32)

    dx = np.diff(array)[1:,:] # remove the first row
    dy = np.diff(array, axis=0)[:,1:] # remove the first column
    dnorm = np.sqrt(dx**2 + dy**2)
    sharpness = np.average(dnorm)
    return sharpness

def calculate_brightness(image_path, rois):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a blank mask
    mask = np.zeros_like(image)

    # Set pixel values inside ROIs to 255 (white)
    for roi in rois:
        center, radius = roi
        cv2.circle(mask, center, radius, 255, -1)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    # Calculate the average pixel intensity of the masked image
    brightness = np.mean(masked_image)

    return brightness

def calculate_snr(image_path, rois):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the average pixel intensity within the ROIs
    signals = []
    for roi in rois:
        center, radius = roi
        x, y = center
        region = image[y-radius:y+radius, x-radius:x+radius]
        signal = np.mean(region)
        signals.append(signal)

    # Calculate the standard deviation of pixel intensities across the entire image
    noise = np.std(image)

    # Calculate the SNR as the ratio of signal to noise
    snr = np.mean(signals) / noise

    return snr

def calculate_cnr(image_path, rois):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the average pixel intensity within the ROIs
    signals = []
    for roi in rois:
        center, radius = roi
        x, y = center
        region = image[y-radius:y+radius, x-radius:x+radius]
        signal = np.mean(region)
        signals.append(signal)

    # Calculate the standard deviation of pixel intensities across the entire image
    noise = np.std(image)

    # Calculate the contrast as the difference between the maximum and minimum signals
    contrast = max(signals) - min(signals)

    # Calculate the CNR as the ratio of contrast to noise
    cnr = contrast / noise

    return cnr

def calculate_dynamic_range(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocessing
    if len(image.shape) > 2:  # Convert color image to grayscale
        image = np.mean(image, axis=2)

    # Calculate minimum and maximum intensities
    min_val = np.min(image)
    max_val = np.max(image)

    # Calculate dynamic range
    dynamic_range = max_val - min_val

    return dynamic_range

def calculate_uniformity(image_path, rois):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the standard deviation of pixel intensities within the ROIs
    intensities = []
    for roi in rois:
        center, radius = roi
        x, y = center
        region = image[y-radius:y+radius, x-radius:x+radius]
        intensities.extend(region.flatten())
    
    uniformity = np.std(intensities)

    return uniformity

def calculate_linearity(image_path, rois):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    linearity_scores = []
    for roi in rois:
        center, radius = roi
        x, y = center
        region = image[y-radius:y+radius, x-radius:x+radius]

        # Calculate the histogram of pixel intensities
        histogram = cv2.calcHist([region], [0], None, [256], [0, 256])
        
        # Calculate the linearity score as the inverse of the histogram entropy
        entropy_value = entropy(histogram, base=2)
        linearity_score = 1.0 - (entropy_value / np.log2(256))
        linearity_scores.append(linearity_score)

    average_linearity = np.mean(linearity_scores)

    return average_linearity

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