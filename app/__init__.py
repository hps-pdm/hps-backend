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

    # Example health endpoint – replace with your real routes
    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # TODO: register blueprints, config, etc. here
    # from .routes import api_bp
    # app.register_blueprint(api_bp)

    return app
