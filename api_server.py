from flask import Flask, jsonify, request
import yaml

from core_ai.reid_matcher import ReIDMatcher
from database.db_handler import DBHandler


def create_app(config_path="config.yaml", db_handler=None, matcher=None):
    app = Flask(__name__)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    db = db_handler or DBHandler(config["database"]["db_path"])
    reid_matcher = matcher or ReIDMatcher(db)

    app.config["APP_CONFIG"] = config
    app.config["DB"] = db
    app.config["REID_MATCHER"] = reid_matcher

    @app.route("/api/status", methods=["GET"])
    def get_status():
        return jsonify({"status": "online", "message": "AI Surveillance Engine is running."})

    @app.route("/api/reassign", methods=["POST"])
    def reassign_stranger():
        data = request.json or {}
        old_id = data.get("old_id")
        new_id = data.get("new_id")
        name = data.get("name", "N/A")
        auth_level = data.get("auth_level", "Verified")

        if not old_id or not new_id:
            return jsonify({"error": "Missing old_id or new_id"}), 400

        db_success = db.reassign_visitor(old_id, new_id, name, auth_level)

        if db_success:
            reid_matcher.update_id_in_memory(old_id, new_id)
            return jsonify({"success": True, "message": f"Successfully reassigned {old_id} to {new_id}"})

        return jsonify({"error": "Database update failed. ID might not exist."}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    api_host = app.config["APP_CONFIG"]["api"]["host"]
    api_port = app.config["APP_CONFIG"]["api"]["port"]
    print(f"[SYSTEM] Starting AI API Server on http://{api_host}:{api_port}")
    app.run(host=api_host, port=api_port, debug=False, use_reloader=False)
