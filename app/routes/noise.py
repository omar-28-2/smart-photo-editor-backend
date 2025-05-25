from flask_restx import Namespace, Resource, fields, reqparse
from flask import request
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
            processed_image_path = save_processed_image(noisy)

            print(f"Original filename: {original_filename}")  # Debug log

            # Debug: Print all logs in database
            all_logs = ImageLog.query.all()
            print("All logs in database:")
            for log in all_logs:
                print(f"  - ID: {log.id}, Filename: {log.filename}, Processed: {log.processed}")

            # Update the existing log entry instead of creating a new one
            existing_log = ImageLog.query.filter_by(filename=original_filename).first()
            print(f"Found existing log: {existing_log}")  # Debug log

            if existing_log:
                print(f"Current processed status: {existing_log.processed}")  # Debug log
                existing_log.processed = True
                try:
                    db.session.commit()
                    print(f"Updated processed status: {existing_log.processed}")  # Debug log
                except Exception as e:
                    print(f"Error committing to database: {str(e)}")  # Debug log
                    db.session.rollback()
                    raise e
            else:
                print("No existing log found, creating new one")  # Debug log
                # Create a new log entry with processed=True since we're applying noise
                new_log = ImageLog(filename=original_filename, processed=True)
                try:
                    db.session.add(new_log)
                    db.session.commit()
                    print(f"Created new log with processed={new_log.processed}")  # Debug log
                except Exception as e:
                    print(f"Error creating new log: {str(e)}")  # Debug log
                    db.session.rollback()
                    raise e

            # Verify the update
            updated_log = ImageLog.query.filter_by(filename=original_filename).first()
            print(f"Final log state: {updated_log}")  # Debug log

            return {
                "message": f"{noise_type} noise added successfully",
                "processed_image": original_filename
            }
        except Exception as e:
            return {"error": str(e)}, 500

@noise_ns.route('/remove')
class RemoveNoise(Resource):
    @noise_ns.expect(remove_noise_model)
    def post(self):
        try:
            image = get_image_from_request(request)
            if image is None:
                return {"error": "No image provided"}, 400

            filter_type = request.form.get('type', 'median')
            params_str = request.form.get('params', '{}')
            
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError:
                return {"error": "Invalid parameters format"}, 400

            if filter_type == 'median':
                denoised = apply_median_filter(image, params.get('kernel_size', 3))
            elif filter_type == 'gaussian':
                denoised = apply_gaussian_filter(image, params.get('kernel_size', 5), params.get('sigma', 0))
            elif filter_type == 'bilateral':
                denoised = apply_bilateral_filter(
                    image,
                    params.get('d', 9),
                    params.get('sigma_color', 75),
                    params.get('sigma_space', 75)
                )
            elif filter_type == 'notch':
                points = params.get('points', None)
                denoised = apply_notch_filter(image, points)
            elif filter_type == 'band_reject':
                denoised = apply_band_reject_filter(
                    image,
                    params.get('cutoff_freq', 30),
                    params.get('width', 10)
                )
            else:
                return {"error": "Invalid filter type"}, 400

            # Get the original filename from the request
            original_filename = request.files['file'].filename
            processed_image_path = save_processed_image(denoised)

            # Update the existing log entry instead of creating a new one
            existing_log = ImageLog.query.filter_by(filename=original_filename).first()
            if existing_log:
                existing_log.processed = True
                db.session.commit()
            else:
                # If no existing log found (shouldn't happen), create a new one
                new_log = ImageLog(filename=original_filename, processed=True)
                db.session.add(new_log)
                db.session.commit()

            return {
                "message": f"Noise removed using {filter_type} filter successfully",
                "processed_image": original_filename
            }
        except Exception as e:
            return {"error": str(e)}, 500
