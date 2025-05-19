from flask import Blueprint, request, jsonify

bp = Blueprint('filters', __name__, url_prefix='/filters')

@bp.route('/apply', methods=['POST'])
def apply_filter():
    return jsonify({"message": "Filter applied (placeholder)"})
