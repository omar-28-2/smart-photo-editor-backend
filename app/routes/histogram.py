from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from app.services.image_io import get_image_from_request, save_processed_image
from app.models.db import db
from app.models.image_log import ImageLog

bp = Blueprint('histogram', __name__, url_prefix='/histogram')

@bp.route('/get', methods=['POST'])
def get_histogram():
    try:
        image = get_image_from_request(request)
        if image is None:
            return jsonify({"error": "No image provided"}), 400

        # Calculate histogram for each channel
        histograms = {}
        for i, col in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histograms[col] = hist.flatten().tolist()

        # Calculate cumulative histogram for each channel
        cumulative_histograms = {}
        for col in histograms:
            cumulative_histograms[col] = np.cumsum(histograms[col]).tolist()

        # Log the histogram calculation
        new_log = ImageLog(filename=request.files['file'].filename, processed=True)
        db.session.add(new_log)
        db.session.commit()

        return jsonify({
            "histograms": histograms,
            "cumulative_histograms": cumulative_histograms,
            "message": "Histogram data retrieved successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/equalize', methods=['POST'])
def equalize_histogram():
    try:
        image = get_image_from_request(request)
        if image is None:
            return jsonify({"error": "No image provided"}), 400

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        merged = cv2.merge((cl, a, b))
        equalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Calculate histograms for both original and equalized images
        original_histograms = {}
        equalized_histograms = {}
        
        for i, col in enumerate(['b', 'g', 'r']):
            # Original histogram
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            original_histograms[col] = hist.flatten().tolist()
            
            # Equalized histogram
            hist = cv2.calcHist([equalized], [i], None, [256], [0, 256])
            equalized_histograms[col] = hist.flatten().tolist()
        
        # Save both original and equalized images
        original_path = save_processed_image(image)
        equalized_path = save_processed_image(equalized)
        
        # Log the histogram equalization
        new_log = ImageLog(filename=equalized_path.split('/')[-1], processed=True)
        db.session.add(new_log)
        db.session.commit()
        
        return jsonify({
            "message": "Histogram equalization completed successfully",
            "original_image": original_path,
            "equalized_image": equalized_path,
            "original_histograms": original_histograms,
            "equalized_histograms": equalized_histograms
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
