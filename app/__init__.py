import os
from flask import Flask, jsonify, request
from app.models.db import db
from app.models.image_log import ImageLog
from sqlalchemy import text
from flask_restx import Api
from flask_cors import CORS

api = Api(
    title="Smart Photo Editor API",
    version="1.0",
    description="API docs for Smart Photo Editor",
    doc="/docs"
)

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    
    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

    db_path = os.path.join(app.instance_path, 'photo_editor.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    os.makedirs(app.instance_path, exist_ok=True)
    db.init_app(app)

    with app.app_context():
        db.create_all()
        
    api.init_app(app)

    from .routes import fft, filters, histogram, mask, noise, upload

    api.add_namespace(fft.fft_ns, path='/fft')
    api.add_namespace(filters.filters_ns, path='/filters')
    api.add_namespace(histogram.hist_ns, path='/histogram')
    api.add_namespace(mask.mask_ns, path='/mask')
    api.add_namespace(noise.noise_ns, path='/noise')
    api.add_namespace(upload.upload_ns, path='/upload')

    @app.route('/')
    def index():
        return jsonify({
            "message": "Welcome to Smart Photo Editor API",
            "endpoints": {
                "test_db": "/test-db",
                "upload": "/upload",
                "noise": {
                    "add": "/noise/add",
                    "remove": "/noise/remove"
                },
                "histogram": {
                    "get": "/histogram/get",
                    "equalize": "/histogram/equalize"
                },
                "filters": "/filters",
                "fft": {
                    "apply": "/fft/apply",
                    "inverse": "/fft/inverse",
                    "magnitude": "/fft/magnitude"
                },
                "image_logs": "/image-logs"
            }
        })

    @app.route('/test-db')
    def test_db():
        try:
            with db.engine.connect() as connection:
                result = connection.execute(text("SELECT 1")).scalar()
            return jsonify({"db_status": "connected", "result": result})
        except Exception as e:
            return jsonify({"db_status": "error", "error": str(e)})

    @app.route('/add-image', methods=['POST'])
    def add_image():
        data = request.json
        filename = data.get('filename')
        if not filename:
            return jsonify({"error": "Filename required"}), 400

        new_log = ImageLog(
            filename=filename,
            processed=False,
            operation="Image added manually"
        )
        db.session.add(new_log)
        db.session.commit()

        return jsonify({"message": "Image log added", "id": new_log.id}), 201

    @app.route('/image-logs')
    def get_image_logs():
        logs = ImageLog.query.all()
        result = []
        for log in logs:
            result.append({
                "id": log.id,
                "filename": log.filename,
                "processed": log.processed
            })
        return jsonify(result)

    @app.route('/fft-info')
    def fft_info():
        return jsonify({
            "fft_endpoints": {
                "apply_fft": "/fft/apply (POST with image file)",
                "inverse_fft": "/fft/inverse (POST with image file)",
                "magnitude_spectrum": "/fft/magnitude (POST with image file)"
            }
        })

    return app
