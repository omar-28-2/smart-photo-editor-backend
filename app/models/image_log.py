from app.models.db import db

class ImageLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    processed = db.Column(db.Boolean, default=False)
    image_data = db.Column(db.Text, nullable=True)