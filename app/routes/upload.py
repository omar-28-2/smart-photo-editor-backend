import os
from flask import current_app, Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.models.db import db
from app.models.image_log import ImageLog

bp = Blueprint("upload", __name__, url_prefix="/upload")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(current_app.root_path, "static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        new_log = ImageLog(filename=filename)
        db.session.add(new_log)
        db.session.commit()

        return (
            jsonify({"message": "Image uploaded and logged", "filename": filename}),
            201,
        )

    return jsonify({"error": "File type not allowed"}), 400


@bp.route("/logs", methods=["GET"])
def get_logs():
    logs = ImageLog.query.all()
    return jsonify(
        [
            {"id": log.id, "filename": log.filename, "processed": log.processed}
            for log in logs
        ]
    )