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
    
    f = fftpack.fft2(image)
    fshift = fftpack.fftshift(f)
    
    rows, cols = image.shape[:2]
    crow, ccol = rows//2, cols//2
    
    mask = np.ones((rows, cols), np.uint8)
    
    if points:
        
        for x, y in points:
            mask[crow-y:crow+y, ccol-x:ccol+x] = 0
    else:
        
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 0
    
    
    fshift = fshift * mask[:, :, np.newaxis]
    
    f_ishift = fftpack.ifftshift(fshift)
    img_back = fftpack.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back.astype(np.uint8)

def apply_band_reject_filter(image, cutoff_freq=30, width=10):
    
    f = fftpack.fft2(image)
    fshift = fftpack.fftshift(f)
    
    rows, cols = image.shape[:2]
    crow, ccol = rows//2, cols//2
    
    
    mask = np.ones((rows, cols), np.uint8)
    r = cutoff_freq
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask[mask_area] = 0
    
    fshift = fshift * mask[:, :, np.newaxis]
    
    f_ishift = fftpack.ifftshift(fshift)
    img_back = fftpack.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back.astype(np.uint8)
