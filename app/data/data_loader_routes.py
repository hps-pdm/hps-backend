"""
Data Loading Web Interface

A simple Flask route that can be added to the main application to trigger
data loading from a web interface.

This provides a convenient way to load data into the cache without using command line.
"""

from flask import Blueprint, render_template_string, request, jsonify, redirect, url_for
import threading
import time
from datetime import datetime
import sys
import os

# Add the project root to the path so we can import the data loader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data_loader import VibrationDataLoader

# Create Blueprint for data loading routes
data_loader_bp = Blueprint('data_loader', __name__, url_prefix='/admin')

# Global variables to track loading status
loading_status = {
    'is_loading': False,
    'current_task': None,
    'progress': 0,
    'total_records': 0,
    'start_time': None,
    'end_time': None,
    'error': None
}

def background_data_load(force=False, days_back=365):
    """Background task to load data."""
    global loading_status
    
    try:
        loading_status.update({
            'is_loading': True,
            'current_task': 'Initializing...',
            'progress': 0,
            'start_time': datetime.now(),
            'error': None
        })
        
        loader = VibrationDataLoader()
        
        # Load equipment metadata
        loading_status['current_task'] = 'Loading equipment metadata...'
        loading_status['progress'] = 10
        equipment_count = loader.load_equipment_metadata(force)
        
        # Load latest vibration data
        loading_status['current_task'] = 'Loading latest vibration data...'
        loading_status['progress'] = 30
        latest_count = loader.load_latest_vibration_data(force)
        
        # Load historical data
        loading_status['current_task'] = 'Loading historical time series data...'
        loading_status['progress'] = 50
        historical_count = loader.load_historical_time_series(force, days_back)
        
        loading_status.update({
            'current_task': 'Complete!',
            'progress': 100,
            'total_records': equipment_count + latest_count + historical_count,
            'end_time': datetime.now(),
            'is_loading': False
        })
        
    except Exception as e:
        loading_status.update({
            'is_loading': False,
            'error': str(e),
            'end_time': datetime.now()
        })

@data_loader_bp.route('/data-loader')
def data_loader_interface():
    """Main data loader interface."""
    
    # Get current cache status
    try:
        loader = VibrationDataLoader()
        cache_stats = loader.get_load_status()
    except:
        cache_stats = []
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Vibration Data Loader</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #1e1e1e; 
            color: #ffffff; 
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .header { 
            background-color: #333; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
        }
        .card { 
            background-color: #2d2d2d; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
            border-left: 4px solid #007bff; 
        }
        .status-card { 
            background-color: #2d2d2d; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 15px; 
        }
        .btn { 
            background-color: #007bff; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
            margin: 5px; 
            font-size: 14px; 
        }
        .btn:hover { 
            background-color: #0056b3; 
        }
        .btn-danger { 
            background-color: #dc3545; 
        }
        .btn-danger:hover { 
            background-color: #c82333; 
        }
        .progress-container { 
            background-color: #444; 
            border-radius: 4px; 
            padding: 3px; 
            margin: 10px 0; 
        }
        .progress-bar { 
            background-color: #28a745; 
            height: 20px; 
            border-radius: 2px; 
            transition: width 0.3s ease; 
        }
        .form-group { 
            margin: 15px 0; 
        }
        .form-group label { 
            display: block; 
            margin-bottom: 5px; 
            font-weight: bold; 
        }
        .form-group input, .form-group select { 
            padding: 8px; 
            border-radius: 4px; 
            border: 1px solid #555; 
            background-color: #444; 
            color: white; 
            width: 200px; 
        }
        .error { 
            color: #dc3545; 
            background-color: #2d1a1a; 
            padding: 10px; 
            border-radius: 4px; 
            margin: 10px 0; 
        }
        .success { 
            color: #28a745; 
            background-color: #1a2d1a; 
            padding: 10px; 
            border-radius: 4px; 
            margin: 10px 0; 
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 10px; 
        }
        th, td { 
            text-align: left; 
            padding: 8px; 
            border-bottom: 1px solid #555; 
        }
        th { 
            background-color: #333; 
        }
        .loading { 
            display: none; 
        }
    </style>
    <script>
        function startLoading(force = false) {
            const daysBack = document.getElementById('days_back').value;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('load-form').style.display = 'none';
            
            fetch('/admin/start-load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    force: force,
                    days_back: parseInt(daysBack)
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started') {
                    checkProgress();
                } else {
                    alert('Failed to start loading: ' + data.message);
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('load-form').style.display = 'block';
                }
            });
        }
        
        function checkProgress() {
            fetch('/admin/load-status')
            .then(response => response.json())
            .then(data => {
                document.getElementById('progress-bar').style.width = data.progress + '%';
                document.getElementById('progress-text').textContent = 
                    data.current_task + ' (' + data.progress + '%)';
                
                if (data.is_loading) {
                    setTimeout(checkProgress, 2000);
                } else {
                    document.getElementById('loading').style.display = 'none';
                    if (data.error) {
                        document.getElementById('result').innerHTML = 
                            '<div class="error">Error: ' + data.error + '</div>';
                    } else {
                        document.getElementById('result').innerHTML = 
                            '<div class="success">Data loading completed! Total records: ' + 
                            data.total_records + '</div>';
                    }
                    setTimeout(() => location.reload(), 3000);
                }
            });
        }
        
        function refreshStatus() {
            location.reload();
        }
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔄 Vibration Data Loader</h1>
            <p>Load vibration analysis data from Databricks into local SQLite cache for improved performance.</p>
        </div>
        
        <div class="card">
            <h3>📊 Current Cache Status</h3>
            {% if cache_stats %}
            <table>
                <tr><th>Data Type</th><th>Records</th><th>Last Updated</th><th>Status</th></tr>
                {% for table_name, last_loaded, record_count, status in cache_stats %}
                <tr>
                    <td>{{ table_name.replace('_', ' ').title() }}</td>
                    <td>{{ "{:,}".format(record_count) }}</td>
                    <td>{{ last_loaded }}</td>
                    <td>{{ status }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No data loaded yet.</p>
            {% endif %}
            <button class="btn" onclick="refreshStatus()">🔄 Refresh Status</button>
        </div>
        
        <div id="load-form" class="card">
            <h3>📥 Load Data</h3>
            <div class="form-group">
                <label for="days_back">Historical Data (Days Back):</label>
                <input type="number" id="days_back" value="365" min="1" max="730">
            </div>
            <button class="btn" onclick="startLoading(false)">
                📥 Load New Data Only
            </button>
            <button class="btn btn-danger" onclick="startLoading(true)">
                🔄 Force Reload All Data
            </button>
            <p style="font-size: 12px; color: #aaa; margin-top: 10px;">
                <strong>Load New Data Only:</strong> Skip loading if data already exists<br>
                <strong>Force Reload:</strong> Delete existing data and reload everything
            </p>
        </div>
        
        <div id="loading" class="card loading">
            <h3>⏳ Loading Data...</h3>
            <div class="progress-container">
                <div id="progress-bar" class="progress-bar" style="width: 0%"></div>
            </div>
            <p id="progress-text">Initializing...</p>
        </div>
        
        <div id="result"></div>
        
        <div class="card">
            <h3>ℹ️ Information</h3>
            <p><strong>What this does:</strong></p>
            <ul>
                <li>Downloads latest vibration data from Databricks</li>
                <li>Downloads historical time-series data for all equipment</li>
                <li>Stores data in local SQLite database for fast access</li>
                <li>Enables offline operation of the vibration analysis app</li>
            </ul>
            <p><strong>Performance Impact:</strong></p>
            <ul>
                <li>Initial load may take several minutes</li>
                <li>Subsequent app queries will be 100-1000x faster</li>
                <li>Reduces dependency on Databricks for routine operations</li>
            </ul>
        </div>
    </div>
</body>
</html>
    """
    
    return render_template_string(html_template, cache_stats=cache_stats)

@data_loader_bp.route('/start-load', methods=['POST'])
def start_load():
    """Start the data loading process."""
    global loading_status
    
    if loading_status['is_loading']:
        return jsonify({'status': 'error', 'message': 'Loading already in progress'})
    
    try:
        data = request.json
        force = data.get('force', False)
        days_back = data.get('days_back', 365)
        
        # Start background loading
        thread = threading.Thread(target=background_data_load, args=(force, days_back))
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'started', 'message': 'Data loading started'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@data_loader_bp.route('/load-status')
def get_load_status():
    """Get current loading status."""
    return jsonify(loading_status)

@data_loader_bp.route('/cache-stats')
def cache_stats():
    """Get cache statistics."""
    try:
        from app.data.CachedVibExtractor import cached_vibextractor
        stats = cached_vibextractor.get_cache_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

# Function to register the blueprint with the main app
def register_data_loader_routes(app):
    """Register data loader routes with the main Flask app."""
    app.register_blueprint(data_loader_bp)
    return data_loader_bp
