from flask import Blueprint, request, jsonify

bp = Blueprint('upload', __name__, url_prefix='/upload')

@bp.route('/image', methods=['POST'])
def upload_image():
    return jsonify({"message": "Image uploaded (placeholder)"})
