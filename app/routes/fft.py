from flask_restx import Namespace, Resource, fields, reqparse
from flask import request
from app.services.fft_utils import apply_fft, apply_ifft, magnitude_spectrum
import numpy as np
import cv2
import base64

fft_ns = Namespace('fft', description='FFT related operations')

def encode_image_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array.astype(np.uint8))
    encoded = base64.b64encode(buffer).decode('utf-8')
    return encoded

file_upload_parser = reqparse.RequestParser()
file_upload_parser.add_argument('image', location='files', type='FileStorage', required=False)
file_upload_parser.add_argument('fft_image', location='files', type='FileStorage', required=False)

@fft_ns.route('/apply')
class FFTApply(Resource):
    @fft_ns.expect(file_upload_parser)
    def post(self):
        args = file_upload_parser.parse_args()
        file = args.get('image')
        if not file:
            return {"error": "No image file provided"}, 400
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        fshift = apply_fft(img)
        mag_spec = magnitude_spectrum(fshift)
        mag_spec_norm = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX)
        encoded_img = encode_image_to_base64(mag_spec_norm)
        return {"magnitude_spectrum": encoded_img}

@fft_ns.route('/inverse')
class FFTInverse(Resource):
    @fft_ns.expect(file_upload_parser)
    def post(self):
        args = file_upload_parser.parse_args()
        file = args.get('fft_image')
        if not file:
            return {"error": "No FFT image file provided"}, 400
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        fshift = apply_ifft(img)
        encoded_img = encode_image_to_base64(fshift)
        return {"inverse_fft": encoded_img}

@fft_ns.route('/magnitude')
class FFTMagnitude(Resource):
    @fft_ns.expect(file_upload_parser)
    def post(self):
        args = file_upload_parser.parse_args()
        file = args.get('fft_image')
        if not file:
            return {"error": "No FFT image file provided"}, 400
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        mag_spec = magnitude_spectrum(img)
        mag_spec_norm = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX)
        encoded_img = encode_image_to_base64(mag_spec_norm)
        return {"magnitude_spectrum": encoded_img}
