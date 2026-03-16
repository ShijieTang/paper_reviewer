import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent.parent))
from modular_seg import save_sections, segment_md, user_update_sec

app = Flask(__name__)

# In-memory state
current_md_name = None
current_sections = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/segments", methods=["POST"])
def segments():
    global current_md_name, current_sections
    data = request.get_json()
    md_name = data.get("md_name", "")
    md_path = data.get("md_path", "data/md")
    try:
        current_sections = segment_md(md_name, md_path=md_path)
        current_md_name = md_name
        save_sections(md_name, current_sections, suffix="raw")
        return jsonify({"sections": [
            {"header": header, "preview": content[:100]}
            for header, content in current_sections.items()
        ]})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/save", methods=["POST"])
def save():
    data = request.get_json()
    responses: dict = data.get("responses", {})
    updated = user_update_sec(current_sections, responses)
    save_sections(current_md_name, updated, suffix="reviewed")
    return jsonify({"message": f"Saved {len(updated)} section(s)."})


if __name__ == "__main__":
    app.run(debug=True)
