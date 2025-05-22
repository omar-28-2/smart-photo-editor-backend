import cv2
import numpy as np
from scipy import fftpack



def add_salt_pepper_noise(image, density=0.05):
    noisy = image.copy()
    
    num_salt = np.ceil(density * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255
    
    num_pepper = np.ceil(density * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0
    return noisy

def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    return cv2.add(image, gaussian)

def add_periodic_noise(image, frequency=20, amplitude=50, pattern='sine'):
    rows, cols = image.shape[:2]
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    X, Y = np.meshgrid(x, y)
    
    if pattern == 'sine':
        noise = amplitude * np.sin(2 * np.pi * frequency * X / cols)
    elif pattern == 'cosine':
        noise = amplitude * np.cos(2 * np.pi * frequency * X / cols)
    else:
        noise = amplitude * np.sin(2 * np.pi * frequency * X / cols)
    
    noise = noise.astype(np.uint8)
    noisy = cv2.add(image, noise[:, :, np.newaxis])
    return noisy
