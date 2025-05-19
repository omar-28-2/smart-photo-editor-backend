import cv2
import numpy as np
from flask import request, current_app
import os
from datetime import datetime
from app.models.db import db
from app.models.image_log import ImageLog

def get_image_from_request(request):
    if 'file' not in request.files:
        return None
    file = request.files['file']
    if file.filename == '':
        return None
    
    # Read image file
    nparr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Log the image processing
    if image is not None:
        filename = file.filename
        new_log = ImageLog(filename=filename, processed=True)
        db.session.add(new_log)
        db.session.commit()
    
    return image

def save_processed_image(image):
    # Create processed directory if it doesn't exist
    upload_folder = os.path.join(current_app.root_path, "static", "processed")
    os.makedirs(upload_folder, exist_ok=True)
    
    # Generate unique filename
    filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(upload_folder, filename)
    
    # Save image
    cv2.imwrite(filepath, image)
    
    # Log the processed image
    new_log = ImageLog(filename=filename, processed=True)
    db.session.add(new_log)
    db.session.commit()
    
    return f"/static/processed/{filename}"

def load_image(file):
    # Load image from file path
    if os.path.exists(file):
        return cv2.imread(file)
    return None

def save_image(image, filename):
    # Save image to file
    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    cv2.imwrite(filepath, image)
    
    # Log the image save
    new_log = ImageLog(filename=filename, processed=True)
    db.session.add(new_log)
    db.session.commit()
    
    return filepath
