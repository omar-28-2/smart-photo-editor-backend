from flask_restx import Namespace, Resource, fields, reqparse
from flask import request
import cv2
import numpy as np
from app.services.image_io import get_image_from_request, save_processed_image
from app.models.db import db
from app.models.image_log import ImageLog

hist_ns = Namespace('histogram', description='Histogram related operations')

# Parser for file upload (multipart/form-data)
file_upload_parser = reqparse.RequestParser()
file_upload_parser.add_argument('file', location='files', type='FileStorage', required=True, help='Image file')

# Models for Swagger responses
histogram_model = hist_ns.model('Histogram', {
    'b': fields.List(fields.Float, description='Blue channel histogram'),
    'g': fields.List(fields.Float, description='Green channel histogram'),
    'r': fields.List(fields.Float, description='Red channel histogram'),
})

histogram_response_model = hist_ns.model('HistogramResponse', {
    'histograms': fields.Nested(histogram_model),
    'cumulative_histograms': fields.Nested(histogram_model),
    'message': fields.String
})

equalize_response_model = hist_ns.model('EqualizeResponse', {
    'message': fields.String,
    'original_image': fields.String,
    'equalized_image': fields.String,
    'original_histograms': fields.Nested(histogram_model),
    'equalized_histograms': fields.Nested(histogram_model)
})

@hist_ns.route('/get')
class GetHistogram(Resource):
    @hist_ns.expect(file_upload_parser)
    @hist_ns.marshal_with(histogram_response_model)
    def post(self):
        try:
            args = file_upload_parser.parse_args()
            file = args.get('file')
            if not file:
                return {"error": "No image provided"}, 400

            image = get_image_from_request(request)
            if image is None:
                return {"error": "Invalid image"}, 400

            histograms = {}
            for i, col in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms[col] = hist.flatten().tolist()

            cumulative_histograms = {}
            for col in histograms:
                cumulative_histograms[col] = np.cumsum(histograms[col]).tolist()

            # Log the histogram calculation
            new_log = ImageLog(filename=file.filename, processed=True)
            db.session.add(new_log)
            db.session.commit()

            return {
                "histograms": histograms,
                "cumulative_histograms": cumulative_histograms,
                "message": "Histogram data retrieved successfully"
            }
        except Exception as e:
            return {"error": str(e)}, 500


@hist_ns.route('/equalize')
class EqualizeHistogram(Resource):
    @hist_ns.expect(file_upload_parser)
    @hist_ns.marshal_with(equalize_response_model)
    def post(self):
        try:
            args = file_upload_parser.parse_args()
            file = args.get('file')
            if not file:
                return {"error": "No image provided"}, 400

            image = get_image_from_request(request)
            if image is None:
                return {"error": "Invalid image"}, 400

            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)

            merged = cv2.merge((cl, a, b))
            equalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

            original_histograms = {}
            equalized_histograms = {}

            for i, col in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                original_histograms[col] = hist.flatten().tolist()

                hist = cv2.calcHist([equalized], [i], None, [256], [0, 256])
                equalized_histograms[col] = hist.flatten().tolist()

            original_path = save_processed_image(image)
            equalized_path = save_processed_image(equalized)

            new_log = ImageLog(filename=equalized_path.split('/')[-1], processed=True)
            db.session.add(new_log)
            db.session.commit()

            return {
                "message": "Histogram equalization completed successfully",
                "original_image": original_path,
                "equalized_image": equalized_path,
                "original_histograms": original_histograms,
                "equalized_histograms": equalized_histograms
            }
        except Exception as e:
            return {"error": str(e)}, 500
