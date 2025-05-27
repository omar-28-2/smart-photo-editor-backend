import cv2
import numpy as np
from scipy import fftpack
from app.services.fft_utils import apply_fft, apply_ifft, magnitude_spectrum


def apply_sobel_filter(image, direction='both', kernel_size=3):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if direction == 'x':
        filtered = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    elif direction == 'y':
        filtered = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    else:  
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        filtered = np.sqrt(sobelx**2 + sobely**2)


    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    
    if len(image.shape) == 3:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    
    return filtered

def apply_laplace_filter(image, kernel_size=3):   
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    filtered = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if len(image.shape) == 3:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    
    return filtered

def apply_gaussian_filter(image, kernel_size=5, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_mean_filter(image, kernel_size=5):
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_sharpen_filter(image, kernel_size=3, strength=1.0):
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]]) * strength
    
    return cv2.filter2D(image, -1, kernel)

def apply_emboss_filter(image, direction='north'):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    kernels = {
        'north': np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]]),
        'south': np.array([[ 1,  1,  1],
                          [ 0,  0,  0],
                          [-1, -1, -1]]),
        'east': np.array([[-1,  0,  1],
                         [-1,  0,  1],
                         [-1,  0,  1]]),
        'west': np.array([[ 1,  0, -1],
                         [ 1,  0, -1],
                         [ 1,  0, -1]])
    }

    
    kernel = kernels.get(direction.lower(), kernels['north'])
    filtered = cv2.filter2D(gray, -1, kernel)
    
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if len(image.shape) == 3:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    
    return filtered

def apply_notch_filter(image, points=None):
    """
    Apply a notch filter to remove periodic noise from an image.
    
    Args:
        image: Input image (grayscale or color)
        points: List of (x,y) coordinates in normalized [-1,1] range indicating noise frequencies
    
    Returns:
        Filtered image and magnitude spectrum for visualization
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply FFT
    fshift = apply_fft(gray)
    
    # Get magnitude spectrum for visualization before filtering
    magnitude_before = magnitude_spectrum(fshift)
    magnitude_before_norm = cv2.normalize(magnitude_before, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Get image dimensions and center
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    
    # Create mask for notch filter
    mask = np.ones((rows, cols), np.float32)
    
    if points and len(points) > 0:
        # Apply notch filter at each specified point
        for x, y in points:
            # Convert normalized coordinates to pixel coordinates
            x_coord = int(ccol + (x * ccol))
            y_coord = int(crow - (y * crow))
            
            # Create circular mask around the point
            r = max(5, int(min(rows, cols) * 0.02))  # Adaptive radius based on image size
            y_coords, x_coords = np.ogrid[:rows, :cols]
            
            # Create smooth transition for the notch using Gaussian
            dist_from_point = np.sqrt((x_coords - x_coord)**2 + (y_coords - y_coord)**2)
            mask_component = 1 - np.exp(-(dist_from_point**2)/(2 * r**2))
            mask = mask * mask_component
            
            # Apply to symmetric point (for real images)
            x_sym = int(ccol - (x * ccol))
            y_sym = int(crow + (y * crow))
            dist_from_symmetric = np.sqrt((x_coords - x_sym)**2 + (y_coords - y_sym)**2)
            mask_component = 1 - np.exp(-(dist_from_symmetric**2)/(2 * r**2))
            mask = mask * mask_component
    else:
        # If no points specified, create a default notch at the center
        r = max(5, int(min(rows, cols) * 0.01))
        y_coords, x_coords = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x_coords - ccol)**2 + (y_coords - crow)**2)
        mask = 1 - np.exp(-(dist_from_center**2)/(2 * r**2))
    
    # Apply mask to FFT
    fshift_filtered = fshift * mask
    
    # Get magnitude spectrum after filtering for visualization
    magnitude_after = magnitude_spectrum(fshift_filtered)
    magnitude_after_norm = cv2.normalize(magnitude_after, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply inverse FFT
    f_ishift = apply_ifft(fshift_filtered)
    img_back = np.abs(f_ishift)
    
    # Normalize and convert back to uint8
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert back to color if input was color
    if len(image.shape) == 3:
        img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    
    return img_back, magnitude_before_norm, magnitude_after_norm

def apply_band_reject_filter(image, cutoff_freq=30, width=10):
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    
    fshift = apply_fft(gray)
    
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    
    mask = np.ones((rows, cols), np.float32)
    r = cutoff_freq
    w = width
    

    y, x = np.ogrid[:rows, :cols]
    center = [crow, ccol]
    
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = np.ones_like(dist_from_center)
    mask[dist_from_center < r - w/2] = 0
    mask[dist_from_center > r + w/2] = 0
    
    fshift = fshift * mask
    
    f_ishift = apply_ifft(fshift)
    img_back = apply_ifft(f_ishift)
    img_back = np.abs(img_back)
    
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if len(image.shape) == 3:
        img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    
    return img_back

def remove_periodic_noise(image, frequency=20, bandwidth=5):
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply FFT
    fshift = apply_fft(gray)
    
    # Get magnitude spectrum for visualization
    magnitude_spectrum = magnitude_spectrum(fshift)
    magnitude_spectrum_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Get image dimensions and center
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    
    # Create mask for periodic noise
    mask = np.ones((rows, cols), np.float32)
    y, x = np.ogrid[:rows, :cols]
    
    # Convert frequency to pixel distance in frequency domain
    # The higher the frequency, the further from center in FFT
    freq_radius = (frequency * min(rows, cols)) / 100  # Convert frequency to radius
    
    # Create circular masks at the noise frequency locations
    # We look in horizontal and vertical directions from the center
    for angle in [0, 90, 180, 270]:  # Look in all four directions
        angle_rad = np.deg2rad(angle)
        center_x = int(ccol + freq_radius * np.cos(angle_rad))
        center_y = int(crow + freq_radius * np.sin(angle_rad))
        
        # Calculate distance from this point
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        # Create smooth transition in the mask using Gaussian
        mask_component = 1 - np.exp(-(dist**2)/(2 * bandwidth**2))
        mask = mask * mask_component
    
    # Ensure mask values are in [0, 1]
    mask = np.clip(mask, 0, 1)
    
    # Apply mask to FFT
    fshift_filtered = fshift * mask
    
    # Inverse FFT
    f_ishift = apply_ifft(fshift_filtered)
    img_back = apply_ifft(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize and convert back to uint8
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert back to color if input was color
    if len(image.shape) == 3:
        img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    
    return img_back, magnitude_spectrum_norm  
