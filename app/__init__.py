from flask import Flask
from flask_cors import CORS

# Make Flask-Compress optional – won't crash if missing in Azure
try:
    from flask_compress import Compress
except ImportError:
    class Compress:
        def __init__(self, app=None):
            if app is not None:
                self.init_app(app)
        def init_app(self, app):
            pass

def create_app() -> Flask:
    app = Flask(__name__)

    # CORS: allow all origins on /api/*
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Compression (real or no-op)
    Compress(app)

    @app.after_request
    def add_cors_headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return resp

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    from .api import api as api_bp
    app.register_blueprint(api_bp, url_prefix="/api")
    return app
