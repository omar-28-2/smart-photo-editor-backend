from app import create_app
from app.models.db import db
from app.models.image_log import ImageLog

app = create_app()

with app.app_context():
    db.create_all()
    print("Database initialized!")
