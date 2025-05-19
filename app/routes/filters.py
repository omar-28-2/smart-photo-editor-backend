from flask import Blueprint, request, jsonify
from app.services.image_io import get_image_from_request, save_processed_image
from app.services.filters import (
    apply_sobel_filter, apply_laplace_filter,
    apply_gaussian_filter, apply_mean_filter,
    apply_median_filter, apply_bilateral_filter,
    apply_sharpen_filter, apply_emboss_filter
)
from app.models.db import db
from app.models.image_log import ImageLog

bp = Blueprint('filters', __name__, url_prefix='/filters')

@bp.route('/apply', methods=['POST'])
def apply_filter():
    try:
        image = get_image_from_request(request)
        if image is None:
            return jsonify({"error": "No image provided"}), 400

        data = request.get_json()
        filter_type = data.get('type', 'sobel')
        params = data.get('params', {})

        if filter_type == 'sobel':
            filtered = apply_sobel_filter(
                image,
                params.get('direction', 'both'),
                params.get('kernel_size', 3)
            )
        elif filter_type == 'laplace':
            filtered = apply_laplace_filter(
                image,
                params.get('kernel_size', 3)
            )
        elif filter_type == 'gaussian':
            filtered = apply_gaussian_filter(
                image,
                params.get('kernel_size', 5),
                params.get('sigma', 0)
            )
        elif filter_type == 'mean':
            filtered = apply_mean_filter(
                image,
                params.get('kernel_size', 5)
            )
        elif filter_type == 'median':
            filtered = apply_median_filter(
                image,
                params.get('kernel_size', 5)
            )
        elif filter_type == 'bilateral':
            filtered = apply_bilateral_filter(
                image,
                params.get('d', 9),
                params.get('sigma_color', 75),
                params.get('sigma_space', 75)
            )
        elif filter_type == 'sharpen':
            filtered = apply_sharpen_filter(
                image,
                params.get('kernel_size', 3),
                params.get('strength', 1.0)
            )
        elif filter_type == 'emboss':
            filtered = apply_emboss_filter(
                image,
                params.get('direction', 'north')
            )
        else:
            return jsonify({"error": "Invalid filter type"}), 400

        processed_image_path = save_processed_image(filtered)
        
        # Log the filter operation
        new_log = ImageLog(filename=processed_image_path.split('/')[-1], processed=True)
        db.session.add(new_log)
        db.session.commit()

        return jsonify({
            "message": f"{filter_type} filter applied successfully",
            "processed_image": processed_image_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
