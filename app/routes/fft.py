from flask_restx import Namespace, Resource, fields, reqparse
from flask import request, current_app
from app.services.fft_utils import apply_fft, apply_ifft, magnitude_spectrum
from app.services.image_io import get_image_from_request, save_processed_image, load_image
import numpy as np
import cv2
import base64
import os
import pickle
import json
from datetime import datetime
import uuid

fft_ns = Namespace('fft', description='FFT related operations')

def encode_image_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array.astype(np.uint8))
    encoded = base64.b64encode(buffer).decode('utf-8')
    return encoded

@fft_ns.route('/apply')
class FFTApply(Resource):
    def post(self):
        try:
            # Initialize variables
            image = None
            filename = None
            
            current_app.logger.info("FFT Apply endpoint called")
            
            # Check if we have JSON data with a filename
            if request.is_json:
                data = request.get_json()
                current_app.logger.info(f"Received JSON data: {data}")
                if data and 'filename' in data:
                    filename = data['filename']
                    current_app.logger.info(f"Loading image from filename: {filename}")
                    
                    # Load the specified image from uploads directory
                    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
                    filepath = os.path.join(upload_folder, filename)
                    current_app.logger.info(f"Full filepath: {filepath}")
                    
                    if not os.path.exists(filepath):
                        current_app.logger.error(f"File not found: {filepath}")
                        return {"error": "Image file not found"}, 404
                    
                    image = cv2.imread(filepath)
                    if image is None:
                        current_app.logger.error("Failed to load image from file")
                        return {"error": "Failed to load image"}, 400
            
            # If no JSON or no image loaded yet, try form data
            if image is None:
                current_app.logger.info("Attempting to get image from form data")
                image = get_image_from_request(request)
                if image is None:
                    current_app.logger.error("No image provided in form data")
                    return {"error": "No image provided"}, 400
                if 'file' in request.files:
                    filename = request.files['file'].filename
                else:
                    # Generate a temporary filename if none provided
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    filename = f"temp_{timestamp}_{unique_id}.png"
                current_app.logger.info(f"Got image, using filename: {filename}")

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            current_app.logger.info("Applying FFT")
            # Apply FFT and get magnitude spectrum
            fshift = apply_fft(gray)
            mag_spec = magnitude_spectrum(fshift)
            mag_spec_norm = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX)
            
            # Save the FFT image (magnitude spectrum for visualization)
            upload_folder = os.path.join(current_app.root_path, "static", "uploads")
            os.makedirs(upload_folder, exist_ok=True)
            
            # Generate filenames based on input filename
            base_filename = os.path.splitext(filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            fft_filename = f"fft_{base_filename}_{timestamp}_{unique_id}.png"
            fft_data_filename = f"fft_data_{base_filename}_{timestamp}_{unique_id}.pkl"
            
            current_app.logger.info(f"Saving FFT files: {fft_filename}, {fft_data_filename}")
            
            # Save magnitude spectrum image
            filepath = os.path.join(upload_folder, fft_filename)
            cv2.imwrite(filepath, mag_spec_norm)
            
            # Save actual FFT data for later use in filtering
            fft_data_path = os.path.join(upload_folder, fft_data_filename)
            with open(fft_data_path, 'wb') as f:
                pickle.dump(fshift, f)
            
            return {
                "message": "FFT generated successfully",
                "fft_image": fft_filename,
                "fft_data": fft_data_filename
            }
        except Exception as e:
            current_app.logger.error(f"Error in FFT generation: {str(e)}")
            current_app.logger.exception("Full traceback:")
            return {"error": str(e)}, 500

@fft_ns.route('/inverse')
class FFTInverse(Resource):
    def post(self):
        try:
            # Initialize variables
            image = None
            filename = None
            
            current_app.logger.info("FFT Inverse endpoint called")
            
            # Check if we have JSON data
            if request.is_json:
                data = request.get_json()
                current_app.logger.info(f"Received JSON data: {data}")
                if data and 'filename' in data:
                    filename = data['filename']
                    current_app.logger.info(f"Loading image from filename: {filename}")
                    
                    # Load the specified image
                    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
                    filepath = os.path.join(upload_folder, filename)
                    current_app.logger.info(f"Full filepath: {filepath}")
                    
                    if not os.path.exists(filepath):
                        current_app.logger.error(f"File not found: {filepath}")
                        return {"error": "Image file not found"}, 404
                    
                    image = cv2.imread(filepath)
                    if image is None:
                        current_app.logger.error("Failed to load image from file")
                        return {"error": "Failed to load image"}, 400
            
            # If no JSON or no image loaded yet, try form data
            if image is None:
                current_app.logger.info("Attempting to get image from form data")
                image = get_image_from_request(request)
                if image is None:
                    current_app.logger.error("No image provided in form data")
                    return {"error": "No image provided"}, 400
                filename = request.files['file'].filename
                current_app.logger.info(f"Got image from form data, filename: {filename}")

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Get the FFT data filename
            base_filename = os.path.splitext(filename)[0]
            # Look for the most recent FFT data file for this image
            upload_folder = os.path.join(current_app.root_path, "static", "uploads")
            fft_data_files = [f for f in os.listdir(upload_folder) if f.startswith(f"fft_data_{base_filename}_") and f.endswith('.pkl')]
            
            if not fft_data_files:
                current_app.logger.error("No FFT data found for this image")
                return {"error": "FFT data not found"}, 404
                
            # Get the most recent FFT data file
            fft_data_filename = sorted(fft_data_files)[-1]
            fft_data_path = os.path.join(upload_folder, fft_data_filename)
            current_app.logger.info(f"Using FFT data file: {fft_data_filename}")
            
            # Load the FFT data
            with open(fft_data_path, 'rb') as f:
                fshift = pickle.load(f)
            
            current_app.logger.info("Applying inverse FFT")
            # Apply inverse FFT
            processed_img = apply_ifft(fshift)
            processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Save the processed image
            processed_filename = save_processed_image(processed_img)
            current_app.logger.info(f"Saved processed image as: {processed_filename}")
            
            return {
                "message": "Inverse FFT applied successfully",
                "processed_image": processed_filename
            }
        except Exception as e:
            current_app.logger.error(f"Error in inverse FFT: {str(e)}")
            current_app.logger.exception("Full traceback:")
            return {"error": str(e)}, 500

@fft_ns.route('/magnitude')
class FFTMagnitude(Resource):
    def post(self):
        try:
            data = request.get_json()
            if data and 'filename' in data:
                # Load the specified image
                filename = data['filename']
                upload_folder = os.path.join(current_app.root_path, "static", "uploads")
                filepath = os.path.join(upload_folder, filename)
                
                if not os.path.exists(filepath):
                    return {"error": "Image file not found"}, 404
                    
                image = cv2.imread(filepath)
                if image is None:
                    return {"error": "Failed to load image"}, 400
            else:
                # Get image from request
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
            current_app.logger.error(f"Error in magnitude spectrum: {str(e)}")
            return {"error": str(e)}, 500
