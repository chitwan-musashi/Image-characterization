import cv2
import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import scipy
import math

def measure_sharpness(image_path):
    # Load the image
    image = cv2.imread(image_path, 0)  # Load as grayscale

    # Calculate the Laplacian variance as a measure of sharpness
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = np.var(laplacian)

    # Return the sharpness measure
    return variance

def measure_contrast(image_path):
    # Convert the image to grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Compute the histogram of the grayscale image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # Calculate contrast as the standard deviation of the histogram
    contrast = np.std(hist)
    return contrast

def measure_brightness(image_path):
    # Convert the image to grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Calculate the average pixel value as a measure of brightness
    brightness = np.mean(gray)
    return brightness

def contrast_enhancement(gray_image):
    
    # Compute min and max intensity values
    min_val = np.min(gray_image)
    max_val = np.max(gray_image)
    
    # Compute dynamic range
    range_val = max_val - min_val
    
    # Set desired output range
    min_out = 0
    max_out = 255
    
    # Apply contrast enhancement
    enhanced_image = ((gray_image - min_val) / range_val) * (max_out - min_out) + min_out
    
    # Clip values to [min_out, max_out]
    enhanced_image = np.clip(enhanced_image, min_out, max_out)
    
    # Convert back to uint8
    enhanced_image = enhanced_image.astype(np.uint8)
    
    return enhanced_image

def generate_edge_map(gray_image, threshold1, threshold2):
    
    # Apply Gaussian smoothing
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1, threshold2)

    return edges

def detect_edge_points(edge_map):
    # Find contours in the edge map
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edge_points = []

    # Iterate over detected contours
    for contour in contours:
        # Iterate over points in the contour
        for point in contour:
            x, y = point[0]
            edge_points.append((x, y))

    return edge_points

def extract_line_profile(edge_map, point, sampling_width):
    # Compute gradient magnitude and direction using the Sobel operator
    gradient_x = cv2.Sobel(edge_map, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(edge_map, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Select the edge point and its gradient direction
    x, y = point
    orientation = gradient_direction[y, x]

    # Define line profile
    half_width = int(sampling_width / 2)
    x_start = x - int(half_width * np.cos(orientation))
    x_end = x + int(half_width * np.cos(orientation))
    y_start = y - int(half_width * np.sin(orientation))
    y_end = y + int(half_width * np.sin(orientation))

    # Convert gradient magnitude to uint8
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    # Extract intensity values along the line profile
    line_profile = cv2.line(gradient_magnitude, (x_start, y_start), (x_end, y_end), 0, 1)
    line_profile = line_profile[y_start:y_end + 1, x_start:x_end + 1]

    return line_profile

def compute_esf(line_profile):
    '''if line_profile.size == 0:
        raise ValueError('Line profile is empty.')'''

    # Compute cumulative sum along the line profile
    esf = np.cumsum(line_profile)

    return esf

def compute_lsf(esf):
    '''if esf.size == 0:
        raise ValueError('ESF is empty.')'''

    # Compute LSF by taking the derivative of ESF using central differences
    lsf = np.diff(esf)

    return lsf

def compute_mtf(lsf):
    '''if lsf.size == 0:
        raise ValueError('LSF is empty.')'''

    # Compute MTF by taking the Fourier Transform of LSF
    mtf = np.abs(np.fft.fftshift(np.fft.fft(lsf)))

    return mtf


def calculate_mtf(image_path, sampling_width, edge_point_index):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image if necessary (e.g., noise reduction)
    image = contrast_enhancement(image)

    # Detect edges and create an edge map
    edge_map = generate_edge_map(image, 50, 150)
    
    # Select a point on the edge map
    edge_points = detect_edge_points(edge_map)
   
    if len(edge_points) > 0:
        # Select the first detected edge point
        point = edge_points[edge_point_index]
        
        # Extract line profile
        line_profile = extract_line_profile(edge_map, point, sampling_width)

        # Compute Edge Spread Function (ESF)
        esf = compute_esf(line_profile)

        # Compute Line Spread Function (LSF)
        lsf = compute_lsf(esf)

        # Compute the modulation transfer function (MTF) by taking the Fourier Transform of the LSF
        mtf = compute_mtf(lsf)
     
        # Normalize the MTF
        max_mtf = np.max(mtf)
        if max_mtf != 0:
            mtf /= max_mtf
        else:
            print("Maximum value of MTF is zero. Cannot normalize.")

    else:
        print("No edge points found.")

    # Compute spatial frequency
    freq = np.linspace(-5, 5, len(mtf))

    # Calculate Nyquist Value
    for i in range(len(freq)-1):
        if freq[i] >= 0.4 and freq[i+1] <= 0.6:
            nyquist = (abs(mtf[i+1] - mtf[i]) / (freq[i+1] - freq[i]))*0.5
            return nyquist
'''        
    # Plot MTF
    plt.plot(freq[len(mtf)//2:], mtf[len(mtf)//2:])
    plt.xlabel('Spatial Frequency (cycles/pixel)')
    plt.ylabel('Modulation Transfer Function (MTF)')
    plt.title('MTF Plot')
    plt.grid(True)
    plt.show()

'''


def measure_SNR(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate maximum pixel value (signal range)
    max_pixel_value = np.iinfo(image.dtype).max

    # Calculate signal power
    signal_power = np.mean(image)

    # Calculate noise power
    noise_power = np.std(image)

    # Calculate SNR in linear scale
    snr_linear = (signal_power / noise_power) * (max_pixel_value / signal_power)

    # Convert SNR to decibels
    snr_dB = 10 * math.log10(snr_linear)

    return snr_dB

def measure_CNR(image_path, roi_mask=None):
    image = cv2.imread(image_path)
    # Preprocessing
    if len(image.shape) > 2:  # Convert color image to grayscale
        image = np.mean(image, axis=2)

    # Apply ROI mask if provided
    if roi_mask is not None:
        image = np.where(roi_mask, image, np.nan)

    # Calculate mean and standard deviation
    mean_foreground = np.nanmean(image)
    std_background = np.nanstd(image)

    # Calculate CNR
    cnr = mean_foreground / std_background

    return cnr



def measure_dynamic_range(image_path):
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



def measure_uniformity(image_path, patch_size):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the number of patches
    num_rows = image.shape[0] // patch_size
    num_cols = image.shape[1] // patch_size

    # Initialize a list to store standard deviations
    std_deviations = []

    # Iterate over patches
    for r in range(num_rows):
        for c in range(num_cols):
            # Extract the patch
            patch = image[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size]

            # Calculate the standard deviation of intensity values
            std_dev = np.std(patch)

            # Store the standard deviation
            std_deviations.append(std_dev)

    # Calculate the average standard deviation
    avg_std_dev = np.mean(std_deviations)

    return avg_std_dev


'''
def measure_pixel_defect(image):
'''

def calculate_linearity_score(points, line):
    # Calculate the distances between each point and the line
    distances = np.abs(np.cross(line[:2], points) + line[2]) / np.linalg.norm(line[:2])
    
    # Calculate the linearity score based on the mean squared distance
    linearity_score = np.mean(distances ** 2)
    
    return linearity_score

def measure_linearity(image_path):
    # Load the input image
    image = cv2.imread(image_path)
    
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply the Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    # Calculate linearity score
    linearity_score = 0.0
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            # Perform line fitting using the least squares method
            points = np.column_stack((x1, y1, x2, y2)).reshape(-1, 2)
            ones = np.ones((points.shape[0], 1))
            A = np.hstack((points, ones))
            m, _, _, _ = np.linalg.lstsq(A, -np.ones((points.shape[0],)), rcond=None)
            
            # Calculate linearity score based on the least squares fit
            calculated_linearity_score = calculate_linearity_score(points, m)
            
            # Accumulate linearity scores for each line
            linearity_score += calculated_linearity_score
        
        # Average the linearity scores
        linearity_score /= len(lines)
    
    return linearity_score

