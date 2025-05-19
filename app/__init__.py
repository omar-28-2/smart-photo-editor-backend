import os
from flask import Flask, jsonify
from app.models.db import db
from sqlalchemy import text

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    db_path = os.path.join(app.instance_path, 'photo_editor.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    os.makedirs(app.instance_path, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    from .routes import fft, filters, histogram, mask, noise, upload
    app.register_blueprint(fft.bp)
    app.register_blueprint(filters.bp)
    app.register_blueprint(histogram.bp)
    app.register_blueprint(mask.bp)
    app.register_blueprint(noise.bp)
    app.register_blueprint(upload.bp)

    @app.route('/test-db')
    def test_db():
        try:
            with db.engine.connect() as connection:
                result = connection.execute(text("SELECT 1")).scalar()
            return jsonify({"db_status": "connected", "result": result})
        except Exception as e:
            return jsonify({"db_status": "error", "error": str(e)})

    return app
