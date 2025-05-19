from flask import Blueprint, request, jsonify
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

bp = Blueprint('noise', __name__, url_prefix='/noise')

@bp.route('/add', methods=['POST'])
def add_noise():
    try:
        image = get_image_from_request(request)
        if image is None:
            return jsonify({"error": "No image provided"}), 400

        data = request.form
        params = data.get('params')
        if params:
            import json
            params = json.loads(params)
        else:
            params = {}
        noise_type = data.get('type', 'salt_pepper')

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
            return jsonify({"error": "Invalid noise type"}), 400

        processed_image_path = save_processed_image(noisy)
        
        # Log the noise addition
        new_log = ImageLog(filename=processed_image_path.split('/')[-1], processed=True)
        db.session.add(new_log)
        db.session.commit()

        return jsonify({
            "message": f"{noise_type} noise added successfully",
            "processed_image": processed_image_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/remove', methods=['POST'])
def remove_noise():
    try:
        image = get_image_from_request(request)
        if image is None:
            return jsonify({"error": "No image provided"}), 400

        data = request.form
        params = data.get('params')
        if params:
            import json
            params = json.loads(params)
        else:
            params = {}
        filter_type = data.get('type', 'median')

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
            points = params.get('points', None)  # List of (x,y) coordinates for manual masking
            denoised = apply_notch_filter(image, points)
        elif filter_type == 'band_reject':
            denoised = apply_band_reject_filter(
                image,
                params.get('cutoff_freq', 30),
                params.get('width', 10)
            )
        else:
            return jsonify({"error": "Invalid filter type"}), 400

        processed_image_path = save_processed_image(denoised)
        
        # Log the noise removal
        new_log = ImageLog(filename=processed_image_path.split('/')[-1], processed=True)
        db.session.add(new_log)
        db.session.commit()

        return jsonify({
            "message": f"Noise removed using {filter_type} filter successfully",
            "processed_image": processed_image_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
