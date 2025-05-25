from flask_restx import Namespace, Resource, fields
from flask import request
from app.services.image_io import get_image_from_request, save_processed_image
from app.services.filters import (
    apply_sobel_filter, apply_laplace_filter,
    apply_gaussian_filter, apply_mean_filter,
    apply_median_filter, apply_bilateral_filter,
    apply_sharpen_filter, apply_emboss_filter
)
from app.models.db import db
from app.models.image_log import ImageLog
import json
import base64

filters_ns = Namespace('filters', description='Image filtering operations')

filter_params = fields.Raw(description="Parameters specific to the filter type")

filter_model = filters_ns.model('FilterApply', {
    'type': fields.String(required=True, description='Filter type (sobel, laplace, gaussian, etc.)'),
    'params': filter_params
})


@filters_ns.route('/apply')
class ApplyFilter(Resource):
    @filters_ns.expect(filter_model)
    def post(self):
        try:
            image = get_image_from_request(request)
            if image is None:
                return {"error": "No image provided"}, 400

            filter_type = request.form.get('type', 'sobel')
            params_str = request.form.get('params', '{}')
            
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError:
                return {"error": "Invalid parameters format"}, 400

            if filter_type == 'sobel':
                filtered = apply_sobel_filter(
                    image,
                    params.get('direction', 'both'),
                    int(params.get('kernel_size', 3))
                )
            elif filter_type == 'laplace':
                filtered = apply_laplace_filter(
                    image,
                    int(params.get('kernel_size', 3))
                )
            elif filter_type == 'gaussian':
                filtered = apply_gaussian_filter(
                    image,
                    int(params.get('kernel_size', 5)),
                    float(params.get('sigma', 0))
                )
            elif filter_type == 'mean':
                filtered = apply_mean_filter(
                    image,
                    int(params.get('kernel_size', 5))
                )
            elif filter_type == 'median':
                filtered = apply_median_filter(
                    image,
                    int(params.get('kernel_size', 5))
                )
            elif filter_type == 'bilateral':
                filtered = apply_bilateral_filter(
                    image,
                    int(params.get('d', 9)),
                    float(params.get('sigma_color', 75)),
                    float(params.get('sigma_space', 75))
                )
            elif filter_type == 'sharpen':
                filtered = apply_sharpen_filter(
                    image,
                    int(params.get('kernel_size', 3)),
                    float(params.get('strength', 1.0))
                )
            elif filter_type == 'emboss':
                filtered = apply_emboss_filter(
                    image,
                    params.get('direction', 'north')
                )
            else:
                return {"error": "Invalid filter type"}, 400

            processed_image_path = save_processed_image(filtered)

            try:
                # Get the original image filename from the request
                original_filename = request.files['file'].filename
                print(f"Original filename: {original_filename}")  # Debug log

                # Find the original image record
                original_log = ImageLog.query.filter_by(filename=original_filename).first()
                print(f"Found original log: {original_log}")  # Debug log

                if original_log:
                    # Update the existing record
                    original_log.filename = processed_image_path.split('/')[-1]
                    original_log.processed = True
                    try:
                        with open(processed_image_path, 'rb') as f:
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                            original_log.image_data = image_data
                    except Exception as e:
                        print(f"Warning: Could not update image data: {str(e)}")
                else:
                    # Create a new record if no original record exists
                    new_log = ImageLog(
                        filename=processed_image_path.split('/')[-1],
                        processed=True
                    )
                    try:
                        with open(processed_image_path, 'rb') as f:
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                            new_log.image_data = image_data
                    except Exception as e:
                        print(f"Warning: Could not add image data: {str(e)}")
                    db.session.add(new_log)

                db.session.commit()
                print("Successfully updated database")  # Debug log

                return {
                    "message": f"{filter_type} filter applied successfully",
                    "processed_image": processed_image_path
                }

            except Exception as e:
                print(f"Error in filter route: {str(e)}")  # Debug log
                db.session.rollback()
                return {"error": f"Failed to process image: {str(e)}"}, 500

        except Exception as e:
            print(f"Error in filter route: {str(e)}")  # Debug log
            return {"error": str(e)}, 500

