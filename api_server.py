from flask import Flask, request, jsonify, Response
import yaml
import threading
from database.db_handler import DBHandler
from core_ai.reid_matcher import ReIDMatcher
# Assuming you have a way to access the AI components globally or pass them

app = Flask(__name__)

# --- 1. LOAD MASTER CONFIG ---
with open("config.yaml", "r") as f: 
    config = yaml.safe_load(f)

# Initialize Core Components
db = DBHandler(config['database']['db_path'])
reid_matcher = ReIDMatcher(db)

# --- 2. THE FRONTEND & BACKEND BRIDGE ENDPOINTS ---

@app.route('/api/status', methods=['GET'])
def get_status():
    """Allows teammates to check if the AI engine is online."""
    return jsonify({"status": "online", "message": "AI Surveillance Engine is running."})

@app.route('/api/reassign', methods=['POST'])
def reassign_stranger():
    """
    Your backend team will send a POST request here to register a stranger.
    Expected JSON payload:
    {
        "old_id": "Stranger_1",
        "new_id": "EMP_001",
        "name": "John Doe",
        "auth_level": "Employee"
    }
    """
    data = request.json
    old_id = data.get('old_id')
    new_id = data.get('new_id')
    name = data.get('name', 'N/A')
    auth_level = data.get('auth_level', 'Verified')

    if not old_id or not new_id:
        return jsonify({"error": "Missing old_id or new_id"}), 400

    # 1. Update the SQLite Database
    db_success = db.reassign_visitor(old_id, new_id, name, auth_level)
    
    # 2. Update the AI's live memory instantly
    if db_success:
        reid_matcher.update_id_in_memory(old_id, new_id)
        return jsonify({"success": True, "message": f"Successfully reassigned {old_id} to {new_id}"})
    else:
        return jsonify({"error": "Database update failed. ID might not exist."}), 500

if __name__ == '__main__':
    # Start the API server on the port defined in config.yaml
    api_host = config['api']['host']
    api_port = config['api']['port']
    print(f"[SYSTEM] Starting AI API Server on http://{api_host}:{api_port}")
    
    # Run the Flask server
    app.run(host=api_host, port=api_port, debug=False, use_reloader=False)