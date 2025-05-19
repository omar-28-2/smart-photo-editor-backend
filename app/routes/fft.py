from flask import Blueprint, request, jsonify
from app.services.fft_utils import apply_fft

bp = Blueprint('fft', __name__, url_prefix='/fft')

@bp.route('/apply', methods=['POST'])
def fft_apply():
    return jsonify({"message": "FFT applied (placeholder)"})
