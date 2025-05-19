from flask import Blueprint, request, jsonify

bp = Blueprint('mask', __name__, url_prefix='/mask')

@bp.route('/apply', methods=['POST'])
def apply_mask():
    return jsonify({"message": "Mask applied (placeholder)"})
