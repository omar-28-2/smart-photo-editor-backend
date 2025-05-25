import os
from flask import current_app, request, send_file
from flask_restx import Namespace, Resource, reqparse, fields
from werkzeug.utils import secure_filename
from app.models.db import db
from app.models.image_log import ImageLog
import base64
from io import BytesIO

upload_ns = Namespace('upload', description='Image upload operations')

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type='FileStorage', required=True, help='Image file to upload')

upload_response = upload_ns.model('UploadResponse', {
    'message': fields.String,
    'filename': fields.String
})


log_item = upload_ns.model('LogItem', {
    'id': fields.Integer,
    'filename': fields.String,
    'processed': fields.Boolean
})


@upload_ns.route('/')
class UploadImage(Resource):
    @upload_ns.marshal_with(upload_response, code=201)
    def post(self):
        if 'file' not in request.files:
            upload_ns.abort(400, "No file part")
            
        file = request.files['file']
        
        if file.filename == "":
            upload_ns.abort(400, "No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Create uploads directory if it doesn't exist
            upload_folder = os.path.join(current_app.root_path, "static", "uploads")
            os.makedirs(upload_folder, exist_ok=True)
            
            # Save the file to the filesystem
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            # Read the file content and convert to base64
            file_content = file.read()
            image_data = base64.b64encode(file_content).decode('utf-8')
            
            # Create new log entry with image data
            new_log = ImageLog(
                filename=filename,
                processed=False,
                image_data=image_data
            )
            db.session.add(new_log)
            db.session.commit()

            return {"message": "Image uploaded and logged", "filename": filename}, 201

        upload_ns.abort(400, "File type not allowed")


@upload_ns.route('/logs')
class UploadLogs(Resource):
    @upload_ns.marshal_list_with(log_item)
    def get(self):
        logs = ImageLog.query.all()
        return logs


@upload_ns.route('/update/<filename>')
class UpdateImage(Resource):
    def post(self, filename):
        if 'file' not in request.files:
            return {"error": "No file part"}, 400
            
        file = request.files['file']
        
        if file.filename == "":
            return {"error": "No selected file"}, 400

        if file and allowed_file(file.filename):
            # Read the file content and convert to base64
            file_content = file.read()
            image_data = base64.b64encode(file_content).decode('utf-8')
            
            # Update existing log entry
            image_log = ImageLog.query.filter_by(filename=filename).first()
            if not image_log:
                return {"error": "Image not found"}, 404
                
            image_log.image_data = image_data
            # Get processed flag from form data, default to True if not provided
            processed = request.form.get('processed', 'true').lower() == 'true'
            image_log.processed = processed
            db.session.commit()

            return {"message": "Image updated successfully"}, 200

        return {"error": "File type not allowed"}, 400


@upload_ns.route('/download/<filename>')
class DownloadImage(Resource):
    def get(self, filename):
        try:
            # Check if the image exists in the database
            image_log = ImageLog.query.filter_by(filename=filename).first()
            if not image_log:
                return {"error": "Image not found"}, 404

            # Check if the image has been processed
            if not image_log.processed:
                return {"error": "No changes have been made to this image"}, 400

            # Get the image data from the database
            if not image_log.image_data:
                return {"error": "Image data not found in database"}, 404

            try:
                # Convert base64 to bytes
                image_bytes = base64.b64decode(image_log.image_data)
            except Exception as e:
                return {"error": f"Failed to decode image data: {str(e)}"}, 500
            
            # Create a BytesIO object
            image_io = BytesIO(image_bytes)
            
            # Determine the mimetype based on the file extension
            ext = filename.rsplit('.', 1)[1].lower()
            mimetype = f'image/{ext}' if ext in ['png', 'jpg', 'jpeg', 'gif'] else 'image/png'
            
            return send_file(
                image_io,
                mimetype=mimetype,
                as_attachment=True,
                download_name=filename
            )

        except Exception as e:
            return {"error": f"Server error: {str(e)}"}, 500
