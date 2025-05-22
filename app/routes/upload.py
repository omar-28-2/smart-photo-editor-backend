import os
from flask import current_app, request
from flask_restx import Namespace, Resource, reqparse, fields
from werkzeug.utils import secure_filename
from app.models.db import db
from app.models.image_log import ImageLog

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
    @upload_ns.expect(upload_parser)
    @upload_ns.marshal_with(upload_response, code=201)
    def post(self):
        args = upload_parser.parse_args()
        file = args.get('file')

        if file.filename == "":
            upload_ns.abort(400, "No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_folder = os.path.join(current_app.root_path, "static", "uploads")
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            new_log = ImageLog(filename=filename)
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
