from flask import Blueprint, request, jsonify
from app.services.fft_utils import apply_fft, apply_ifft, magnitude_spectrum
import numpy as np
import cv2
import base64
import io

bp = Blueprint('fft', __name__, url_prefix='/fft')

def encode_image_to_base64(img_array):
    """
    Convert numpy image array to base64 encoded PNG string.
    """
    _, buffer = cv2.imencode('.png', img_array.astype(np.uint8))
    encoded = base64.b64encode(buffer).decode('utf-8')
    return encoded

@bp.route('/apply', methods=['POST'])
def fft_apply():
    
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image file provided"}), 400
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    fshift = apply_fft(img)

    mag_spec = magnitude_spectrum(fshift)

    mag_spec_norm = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX)

    encoded_img = encode_image_to_base64(mag_spec_norm)

    return jsonify({"magnitude_spectrum": encoded_img})

@bp.route('/inverse', methods=['POST'])
def fft_inverse():
    file = request.files.get('fft_image')
    if not file:
        return jsonify({"error": "No FFT image file provided"}), 400
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    fshift = apply_ifft(img)

    encoded_img = encode_image_to_base64(fshift)

    return jsonify({"inverse_fft": encoded_img})


@bp.route('/magnitude', methods=['POST'])
def fft_magnitude():
    file = request.files.get('fft_image')
    if not file:
        return jsonify({"error": "No FFT image file provided"}), 400
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    mag_spec = magnitude_spectrum(img)

    mag_spec_norm = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX)

    encoded_img = encode_image_to_base64(mag_spec_norm)

    return jsonify({"magnitude_spectrum": encoded_img})