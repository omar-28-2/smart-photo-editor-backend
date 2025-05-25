import cv2
import numpy as np
from scipy import fftpack



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
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply FFT
    f = fftpack.fft2(gray)
    fshift = fftpack.fftshift(f)
    
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    
    # Create notch filter mask
    mask = np.ones((rows, cols), np.float32)
    
    if points:
        # Create notches at specified points
        for x, y in points:
            # Convert relative coordinates (-1 to 1) to image coordinates
            x_coord = int(ccol + (x * ccol))
            y_coord = int(crow - (y * crow))  # Invert y coordinate
            
            # Create a small circular notch
            r = 5  # Radius of the notch
            y_coords, x_coords = np.ogrid[:rows, :cols]
            dist_from_point = np.sqrt((x_coords - x_coord)**2 + (y_coords - y_coord)**2)
            mask[dist_from_point < r] = 0
            
            # Also create a notch at the symmetric point
            x_sym = int(ccol - (x * ccol))
            y_sym = int(crow + (y * crow))
            dist_from_symmetric = np.sqrt((x_coords - x_sym)**2 + (y_coords - y_sym)**2)
            mask[dist_from_symmetric < r] = 0
    else:
        # Default behavior: remove DC component
        r = 5
        y_coords, x_coords = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x_coords - ccol)**2 + (y_coords - crow)**2)
        mask[dist_from_center < r] = 0
    
    # Apply the mask
    fshift = fshift * mask
    
    # Inverse FFT
    f_ishift = fftpack.ifftshift(fshift)
    img_back = fftpack.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize the result
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert back to BGR if needed
    if len(image.shape) == 3:
        img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    
    return img_back

def apply_band_reject_filter(image, cutoff_freq=30, width=10):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply FFT
    f = fftpack.fft2(gray)
    fshift = fftpack.fftshift(f)
    
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    
    # Create band reject mask
    mask = np.ones((rows, cols), np.float32)
    r = cutoff_freq
    w = width
    
    # Create a more sophisticated band reject mask
    y, x = np.ogrid[:rows, :cols]
    center = [crow, ccol]
    
    # Create a band reject mask that removes frequencies in a ring
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = np.ones_like(dist_from_center)
    mask[dist_from_center < r - w/2] = 0
    mask[dist_from_center > r + w/2] = 0
    
    # Apply the mask
    fshift = fshift * mask
    
    # Inverse FFT
    f_ishift = fftpack.ifftshift(fshift)
    img_back = fftpack.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize the result
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert back to BGR if needed
    if len(image.shape) == 3:
        img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    
    return img_back
