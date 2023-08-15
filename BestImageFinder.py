import os
import cv2
import numpy as np

def calculate_ROIs(image_path):
    image = cv2.imread(image_path,  cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding
    threshold_value = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours of bright regions
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image and get bounding rectangles
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bounding_rectangles = []
    min_area_threshold = 10000  # Set the minimum area threshold here

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            bounding_rectangles.append((x, y, w, h))
    
    return bounding_rectangles

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, std_dev = cv2.meanStdDev(gray)
    return std_dev[0][0]

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    return max_val - min_val

def calculate_histogram_balance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=256, range=[0, 256])
    hist_norm = hist / hist.sum()
    hist_cumsum = hist_norm.cumsum()
    hist_balance = hist_cumsum.max() - hist_cumsum.min()
    return hist_balance

def calculate_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1]
    saturation_mean = np.mean(saturation)
    return saturation_mean

def calculate_hot_pixels(image):
    hot_pixels = np.sum(image == 255, axis=2)
    hot_pixels_count = np.sum(hot_pixels > 0)
    return hot_pixels_count

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def evaluate_image_quality(image):
    sharpness = calculate_sharpness(image)
    noise = calculate_noise(image)
    contrast = calculate_contrast(image)
    hist_balance = calculate_histogram_balance(image)
    saturation_mean = calculate_saturation(image)
    hot_pixels_count = calculate_hot_pixels(image)
    brightness = calculate_brightness(image)
    return sharpness, noise, contrast, hist_balance, saturation_mean, hot_pixels_count, brightness

def find_best_image(image_folder):
    image_files = [file for file in os.listdir(image_folder) if any(file.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])]
    best_image = None
    best_quality = float('-inf')
    
    contrast_values = []
    brightness_values = []
    
    for file in image_files:
        image_path = os.path.join(image_folder, file)
        image = cv2.imread(image_path)
        try:
            quality = evaluate_image_quality(image)
            rois = calculate_ROIs(image_path)  # Calculate ROIs for the image
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue
        
        contrast_values.append(quality[2])
        brightness_values.append(quality[6])
        
        average_quality = sum(quality) / len(quality)
        
        if average_quality > best_quality:
            # Check if any ROI intersects with the image
            for roi in rois:
                roi_area = roi[2] * roi[3]
                image_area = image.shape[0] * image.shape[1]
                intersection_area = max(0, min(roi[0] + roi[2], image.shape[1]) - max(roi[0], 0)) * \
                                   max(0, min(roi[1] + roi[3], image.shape[0]) - max(roi[1], 0))
                intersection_ratio = intersection_area / image_area
                
                if intersection_ratio > 0.5:  # If the intersection ratio is above a threshold (e.g., 0.5)
                    best_quality = average_quality
                    best_image = image_path
                    break
    
    # Calculate contrast threshold and brightness threshold
    contrast_threshold = np.percentile(contrast_values, 50)  # Select image with medium contrast
    brightness_threshold = np.percentile(brightness_values, 75)  # Set a threshold for brightness (adjust percentile as needed)
    
    for file in image_files:
        image_path = os.path.join(image_folder, file)
        image = cv2.imread(image_path)
        try:
            quality = evaluate_image_quality(image)
            rois = calculate_ROIs(image_path)  # Calculate ROIs for the image
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue
        
        if quality[2] > contrast_threshold or quality[6] > brightness_threshold:
            continue
        
        average_quality = sum(quality) / len(quality)
        
        # Check if any ROI intersects with the image
        for roi in rois:
            roi_area = roi[2] * roi[3]
            image_area = image.shape[0] * image.shape[1]
            intersection_area = max(0, min(roi[0] + roi[2], image.shape[1]) - max(roi[0], 0)) * \
                               max(0, min(roi[1] + roi[3], image.shape[0]) - max(roi[1], 0))
            intersection_ratio = intersection_area / image_area
            
            if intersection_ratio > 0.1:  # If the intersection ratio is above a threshold (e.g., 0.5)
                if average_quality > best_quality:
                    best_quality = average_quality
                    best_image = image_path
                break
    
    return best_image


