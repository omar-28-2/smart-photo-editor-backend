from flask_restx import Namespace, Resource, fields, reqparse
from flask import request
import cv2
import numpy as np
from app.services.image_io import get_image_from_request, save_processed_image
from app.models.db import db
from app.models.image_log import ImageLog

adjust_ns = Namespace('adjust', description='Image adjustment operations')

file_upload_parser = reqparse.RequestParser()
file_upload_parser.add_argument('file', location='files', type='FileStorage', required=True, help='Image file')

adjust_params = fields.Raw(description="Adjustment parameters")

adjust_model = adjust_ns.model('Adjust', {
    'brightness': fields.Float(required=True, description='Brightness adjustment (-100 to 100)'),
    'contrast': fields.Float(required=True, description='Contrast adjustment (-100 to 100)'),
    'saturation': fields.Float(required=True, description='Saturation adjustment (-100 to 100)')
})

@adjust_ns.route('/apply')
class ApplyAdjustments(Resource):
    @adjust_ns.expect(file_upload_parser, adjust_model)
    def post(self):
        try:
            image = get_image_from_request(request)
            if image is None:
                return {"error": "No image provided"}, 400

            data = request.json
            brightness = data.get('brightness', 0)
            contrast = data.get('contrast', 0)
            saturation = data.get('saturation', 0)

            # Convert brightness and contrast to OpenCV format
            brightness = 1 + (brightness / 100.0)  # Convert to multiplier
            contrast = 1 + (contrast / 100.0)  # Convert to multiplier

            # Apply brightness and contrast
            adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

            # Apply saturation
            if saturation != 0:
                # Convert to HSV
                hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
                # Adjust saturation
                hsv[:, :, 1] = hsv[:, :, 1] * (1 + saturation / 100.0)
                # Clip values
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                # Convert back to BGR
                adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            processed_image_path = save_processed_image(adjusted)

            # Get the original filename from the request
            original_filename = request.files['file'].filename

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
                "message": "Adjustments applied successfully",
                "processed_image": original_filename
            }

        except Exception as e:
            return {"error": str(e)}, 500 