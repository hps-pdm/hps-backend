import os
import sys

# Ensure the directory containing this file is first on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Avoid conflict with any third-party 'app' module already imported
if "app" in sys.modules:
    del sys.modules["app"]

from app import create_app  # this will now load *your* app package

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
