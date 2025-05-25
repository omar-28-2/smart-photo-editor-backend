from flask_restx import Namespace, Resource, fields, reqparse
from flask import request, current_app
from app.services.fft_utils import apply_fft, apply_ifft, magnitude_spectrum
from app.services.image_io import get_image_from_request, save_processed_image
import numpy as np
import cv2
import base64
import os
import pickle

fft_ns = Namespace('fft', description='FFT related operations')

def encode_image_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array.astype(np.uint8))
    encoded = base64.b64encode(buffer).decode('utf-8')
    return encoded

@fft_ns.route('/apply')
class FFTApply(Resource):
    def post(self):
        try:
            # Get image from request using the utility function
            image = get_image_from_request(request)
            if image is None:
                return {"error": "No image provided"}, 400

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply FFT and get magnitude spectrum
            fshift = apply_fft(gray)
            mag_spec = magnitude_spectrum(fshift)
            mag_spec_norm = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX)
            
            # Save the FFT image (magnitude spectrum for visualization)
            upload_folder = os.path.join(current_app.root_path, "static", "uploads")
            os.makedirs(upload_folder, exist_ok=True)
            
            # Generate filenames
            original_filename = request.files['file'].filename
            fft_filename = f"fft_{original_filename}"
            fft_data_filename = f"fft_data_{original_filename}"
            
            # Save magnitude spectrum image
            filepath = os.path.join(upload_folder, fft_filename)
            cv2.imwrite(filepath, mag_spec_norm)
            
            # Save actual FFT data
            fft_data_path = os.path.join(upload_folder, fft_data_filename)
            with open(fft_data_path, 'wb') as f:
                pickle.dump(fshift, f)
            
            return {
                "message": "FFT generated successfully",
                "fft_image": fft_filename,
                "fft_data": fft_data_filename
            }
        except Exception as e:
            return {"error": str(e)}, 500

@fft_ns.route('/inverse')
class FFTInverse(Resource):
    def post(self):
        try:
            # Get image from request using the utility function
            image = get_image_from_request(request)
            if image is None:
                return {"error": "No image provided"}, 400

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Get the original filename to find the FFT data
            original_filename = request.files['file'].filename
            fft_data_filename = f"fft_data_{original_filename}"
            upload_folder = os.path.join(current_app.root_path, "static", "uploads")
            fft_data_path = os.path.join(upload_folder, fft_data_filename)
            
            # Load the FFT data
            with open(fft_data_path, 'rb') as f:
                fshift = pickle.load(f)
            
            # Apply inverse FFT
            processed_img = apply_ifft(fshift)
            processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Save the processed image
            processed_filename = save_processed_image(processed_img)
            
            return {
                "message": "Inverse FFT applied successfully",
                "processed_image": processed_filename
            }
        except Exception as e:
            return {"error": str(e)}, 500

@fft_ns.route('/magnitude')
class FFTMagnitude(Resource):
    def post(self):
        try:
            # Get image from request using the utility function
            image = get_image_from_request(request)
            if image is None:
                return {"error": "No image provided"}, 400

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            mag_spec = magnitude_spectrum(gray)
            mag_spec_norm = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX)
            encoded_img = encode_image_to_base64(mag_spec_norm)
            return {"magnitude_spectrum": encoded_img}
        except Exception as e:
            return {"error": str(e)}, 500
