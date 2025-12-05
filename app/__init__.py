from flask import Flask
from flask_cors import CORS

# Make Flask-Compress optional – won't crash if missing in Azure
try:
    from flask_compress import Compress
except ImportError:
    class Compress:  # no-op fallback
        def __init__(self, app=None):
            if app is not None:
                self.init_app(app)

        def init_app(self, app):
            pass


def create_app() -> Flask:
    """Application factory used by both local dev and Gunicorn."""
    app = Flask(__name__)

    # CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Compression (real or no-op)
    Compress(app)

    # Health endpoint
    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # Register main API blueprint
    try:
        from .api import api as api_bp

        app.register_blueprint(api_bp, url_prefix="/api")
    except Exception as exc:
        # In production we want hard failure to surface missing routes;
        # during early bootstrap you could log instead.
        raise exc

    return app
