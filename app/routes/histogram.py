from flask import Blueprint, request, jsonify

bp = Blueprint('histogram', __name__, url_prefix='/histogram')

@bp.route('/equalize', methods=['POST'])
def equalize_histogram():
    return jsonify({"message": "Histogram equalization done (placeholder)"})
