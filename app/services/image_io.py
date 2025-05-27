import cv2
import numpy as np
from flask import request, current_app
import os
from datetime import datetime
import uuid
from app.models.db import db
from app.models.image_log import ImageLog

def get_image_from_request(request):
    if 'file' not in request.files:
        return None
    file = request.files['file']
    if file.filename == '':
        return None
    
    nparr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is not None:
        # Save the uploaded image first
        upload_folder = os.path.join(current_app.root_path, "static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        cv2.imwrite(filepath, image)
    
    return image

def save_processed_image(image, prefix='processed_'):
    """
    Save a processed image with a unique filename.
    
    Args:
        image: The image array to save
        prefix: Optional prefix for the filename (default: 'processed_')
    
    Returns:
        The filename of the saved image
    """
    # Use the same static folder for all images
    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    
    # Generate a unique filename using timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{prefix}{timestamp}_{unique_id}.png"
    
    # Save the image
    filepath = os.path.join(upload_folder, filename)
    cv2.imwrite(filepath, image)
    
    return filename

def load_image(filename):
    # Get the full path to the uploads directory
    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    filepath = os.path.join(upload_folder, filename)
    
    if os.path.exists(filepath):
        return cv2.imread(filepath)
    return None

def save_image(image, filename):
    # Use the same static folder for all images
    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    cv2.imwrite(filepath, image)
    
    # Update the existing log entry
    existing_log = ImageLog.query.filter_by(filename=filename).first()
    if existing_log:
        existing_log.processed = True
        db.session.commit()
    
    return filename  # Return just the filename, not the path
