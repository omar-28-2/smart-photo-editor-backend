import numpy as np
def apply_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift


def apply_ifft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


def magnitude_spectrum(fshift):
    mg = 20*np.log(np.abs(fshift)+1)
    return mg

