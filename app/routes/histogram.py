from flask_restx import Namespace, Resource, fields, reqparse
from flask import request, current_app
import cv2
import numpy as np
from app.services.image_io import get_image_from_request, save_processed_image, load_image
from app.models.db import db
from app.models.image_log import ImageLog
import logging
import os

hist_ns = Namespace('histogram', description='Histogram related operations')

file_upload_parser = reqparse.RequestParser()
file_upload_parser.add_argument('file', location='files', type='FileStorage', required=False, help='Image file')
file_upload_parser.add_argument('filename', type=str, required=False, help='Image filename')

histogram_model = hist_ns.model('Histogram', {
    'gray': fields.List(fields.Float, description='Grayscale histogram')
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
            # Get the request data
            data = request.get_json()
            if not data or 'filename' not in data:
                return {"error": "No filename provided"}, 400

            filename = data['filename']
            current_app.logger.info(f"Loading image from filename: {filename}")

            # Get the full path to the uploads directory
            upload_folder = os.path.join(current_app.root_path, "static", "uploads")
            filepath = os.path.join(upload_folder, filename)
            current_app.logger.info(f"Full filepath: {filepath}")

            if not os.path.exists(filepath):
                current_app.logger.error(f"File not found: {filepath}")
                return {"error": "Image file not found"}, 404

            # Load and process the image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if image is None:
                current_app.logger.error("Failed to load image")
                return {"error": "Failed to load image"}, 400

            # Calculate histograms
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            histograms = {'gray': hist.flatten().tolist()}

            cumulative_histograms = {'gray': np.cumsum(hist.flatten()).tolist()}

            return {
                "histograms": histograms,
                "cumulative_histograms": cumulative_histograms,
                "message": "Histogram data retrieved successfully"
            }
        except Exception as e:
            current_app.logger.error(f"Error in histogram generation: {str(e)}")
            return {"error": str(e)}, 500

@hist_ns.route('/equalize')
class EqualizeHistogram(Resource):
    @hist_ns.expect(file_upload_parser)
    @hist_ns.marshal_with(equalize_response_model)
    def post(self):
        try:
            # Get the request data
            data = request.get_json()
            if not data or 'filename' not in data:
                return {"error": "No filename provided"}, 400

            filename = data['filename']
            current_app.logger.info(f"Loading image from filename: {filename}")

            # Get the full path to the uploads directory
            upload_folder = os.path.join(current_app.root_path, "static", "uploads")
            filepath = os.path.join(upload_folder, filename)
            current_app.logger.info(f"Full filepath: {filepath}")

            if not os.path.exists(filepath):
                current_app.logger.error(f"File not found: {filepath}")
                return {"error": "Image file not found"}, 404

            # Load and process the image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                current_app.logger.error("Failed to load image")
                return {"error": "Failed to load image"}, 400

            # Apply CLAHE equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(image)

            # Calculate histograms
            original_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            equalized_hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])

            original_histograms = {'gray': original_hist.flatten().tolist()}
            equalized_histograms = {'gray': equalized_hist.flatten().tolist()}

            # Save original and equalized images
            original_path = save_processed_image(image)
            equalized_path = save_processed_image(equalized)

            return {
                "message": "Histogram equalization completed successfully",
                "original_image": original_path,
                "equalized_image": equalized_path,
                "original_histograms": original_histograms,
                "equalized_histograms": equalized_histograms
            }
        except Exception as e:
            current_app.logger.error(f"Error in histogram equalization: {str(e)}")
            return {"error": str(e)}, 500
