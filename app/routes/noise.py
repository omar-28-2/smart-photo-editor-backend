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

# Model for params (flexible, raw JSON)
params_field = fields.Raw(description='Filter or noise specific parameters')

# Model for add noise request
add_noise_model = noise_ns.model('AddNoise', {
    'type': fields.String(required=True, description='Noise type (salt_pepper, gaussian, periodic)'),
    'params': params_field
})

# Model for remove noise request
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

            data = request.json or {}
            noise_type = data.get('type', 'salt_pepper')
            params = data.get('params', {})

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

            processed_image_path = save_processed_image(noisy)

            new_log = ImageLog(filename=processed_image_path.split('/')[-1], processed=True)
            db.session.add(new_log)
            db.session.commit()

            return {
                "message": f"{noise_type} noise added successfully",
                "processed_image": processed_image_path
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

            data = request.json or {}
            filter_type = data.get('type', 'median')
            params = data.get('params', {})

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

            processed_image_path = save_processed_image(denoised)

            new_log = ImageLog(filename=processed_image_path.split('/')[-1], processed=True)
            db.session.add(new_log)
            db.session.commit()

            return {
                "message": f"Noise removed using {filter_type} filter successfully",
                "processed_image": processed_image_path
            }
        except Exception as e:
            return {"error": str(e)}, 500
