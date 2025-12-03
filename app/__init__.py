from flask import Flask
from flask_cors import CORS

# Make Flask-Compress optional so missing package doesn't crash the app
try:
    from flask_compress import Compress
except ImportError:
    class Compress:  # fallback no-op compressor
        def __init__(self, app=None):
            if app is not None:
                self.init_app(app)

        def init_app(self, app):
            # do nothing – no compression available
            pass
