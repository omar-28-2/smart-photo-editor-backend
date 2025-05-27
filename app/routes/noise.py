from flask_restx import Namespace, Resource, fields, reqparse
from flask import request, current_app
from app.services.image_io import get_image_from_request, save_processed_image
from app.services.noise_utils import (
    add_salt_pepper_noise, add_gaussian_noise, add_periodic_noise
)
from app.services.filters import (
    apply_median_filter, apply_gaussian_filter, apply_bilateral_filter,
    apply_notch_filter, apply_band_reject_filter
)
from app.models.db import db
from app.models.image_log import ImageLog
import json
import os
import numpy as np
import cv2

noise_ns = Namespace('noise', description='Noise addition and removal operations')


params_field = fields.Raw(description='Filter or noise specific parameters')


add_noise_model = noise_ns.model('AddNoise', {
    'type': fields.String(required=True, description='Noise type (salt_pepper, gaussian, periodic)'),
    'params': params_field
})


remove_noise_model = noise_ns.model('RemoveNoise', {
    'type': fields.String(required=True, description='Filter type (median, gaussian, bilateral, notch, band_reject)'),
    'params': params_field
})


@noise_ns.route('/add')
class AddNoise(Resource):
    @noise_ns.expect(add_noise_model)
    def post(self):
        try:
            image = get_image_from_request(request)
            if image is None:
                return {"error": "No image provided"}, 400

            noise_type = request.form.get('type', 'salt_pepper')
            params_str = request.form.get('params', '{}')
            
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError:
                return {"error": "Invalid parameters format"}, 400

            if noise_type == 'salt_pepper':
                noisy = add_salt_pepper_noise(image, params.get('density', 0.05))
            elif noise_type == 'gaussian':
                noisy = add_gaussian_noise(image, params.get('mean', 0), params.get('sigma', 25))
            elif noise_type == 'periodic':
                noisy = add_periodic_noise(
                    image,
                    params.get('frequency', 20),
                    params.get('amplitude', 50),
                    params.get('pattern', 'sine')
                )
            else:
                return {"error": "Invalid noise type"}, 400

            # Get the original filename from the request
            original_filename = request.files['file'].filename
            
            # Save the noisy image with a unique filename
            processed_image_filename = save_processed_image(noisy)

            # Update the existing log entry instead of creating a new one
            existing_log = ImageLog.query.filter_by(filename=original_filename).first()
            if existing_log:
                existing_log.processed = True
                db.session.commit()
            else:
                # Create a new log entry with processed=True since we're applying noise
                new_log = ImageLog(filename=original_filename, processed=True)
                db.session.add(new_log)
                db.session.commit()

            return {
                "message": f"{noise_type} noise added successfully",
                "processed_image": processed_image_filename  # Return the new filename
            }
        except Exception as e:
            return {"error": str(e)}, 500

@noise_ns.route('/remove')
class RemoveNoise(Resource):
    def post(self):
        try:
            if 'file' not in request.files:
                return {'error': 'No file provided'}, 400
            
            file = request.files['file']
            if file.filename == '':
                return {'error': 'No file selected'}, 400
            
            filter_type = request.form.get('type')
            params_str = request.form.get('params', '{}')
            
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError:
                return {'error': 'Invalid parameters format'}, 400
            
            # Read the image
            img_array = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return {'error': 'Invalid image format'}, 400
            
            # Apply the selected filter
            if filter_type == 'median':
                kernel_size = params.get('kernel_size', 3)
                filtered_img = apply_median_filter(img, kernel_size)
                processed_filename = save_processed_image(filtered_img)
                return {
                    'message': 'Median filter applied successfully',
                    'processed_image': processed_filename
                }
            elif filter_type == 'notch':
                points = params.get('points', [])
                # Convert points to list of tuples if they exist
                if points:
                    points = [(float(p['x']), float(p['y'])) for p in points]
                filtered_img, fft_before, fft_after = apply_notch_filter(img, points)
                
                # Save all images
                processed_filename = save_processed_image(filtered_img)
                fft_before_filename = save_processed_image(fft_before, prefix='fft_before_')
                fft_after_filename = save_processed_image(fft_after, prefix='fft_after_')
                
                return {
                    'message': 'Notch filter applied successfully',
                    'processed_image': processed_filename,
                    'fft_before': fft_before_filename,
                    'fft_after': fft_after_filename
                }
            elif filter_type == 'band_reject':
                cutoff_freq = params.get('cutoff_freq', 30)
                width = params.get('width', 10)
                filtered_img = apply_band_reject_filter(img, cutoff_freq, width)
                processed_filename = save_processed_image(filtered_img)
                return {
                    'message': 'Band reject filter applied successfully',
                    'processed_image': processed_filename
                }
            else:
                return {'error': 'Invalid filter type'}, 400
            
        except Exception as e:
            current_app.logger.error(f"Error in noise removal: {str(e)}")
            current_app.logger.exception("Full traceback:")
            return {'error': str(e)}, 500
