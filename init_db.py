from app import create_app
from app.models.db import db
from app.models.image_log import ImageLog

def init_db():
    app = create_app()
    with app.app_context():
        # Drop all tables
        db.drop_all()
        # Create all tables
        db.create_all()
        print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()
