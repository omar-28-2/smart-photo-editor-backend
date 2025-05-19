from flask import Blueprint, request, jsonify

bp = Blueprint('noise', __name__, url_prefix='/noise')

@bp.route('/add', methods=['POST'])
def add_noise():
    return jsonify({"message": "Noise added (placeholder)"})
