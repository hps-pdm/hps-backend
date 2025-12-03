from flask import Flask
from flask_compress import Compress
from app.api import api

def create_app():
    app = Flask(__name__)
    Compress(app)
    app.register_blueprint(api, url_prefix="/api")
    return app
