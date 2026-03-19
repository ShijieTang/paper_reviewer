import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent.parent))
from modular_seg import reconstruct_md, save_sections, segment_md

app = Flask(__name__)

# In-memory state
current_md_name = None
current_sections = {}
current_topic = None                          # set by /api/segments, read by /api/save

VALID_TOPICS = {"Machine Learning Algorithm", "NLP", "AI for Science"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/segments", methods=["POST"])
def segments():
    global current_md_name, current_sections, current_topic
    data = request.get_json()
    md_name = data.get("md_name", "")
    md_path = data.get("md_path", "data/md")
    topic   = data.get("topic", "")

    if topic not in VALID_TOPICS:
        return jsonify({"error": f"Invalid topic '{topic}'. Choose from: {sorted(VALID_TOPICS)}"}), 400

    try:
        current_sections = segment_md(md_name, md_path=md_path)
        current_md_name  = md_name
        current_topic    = topic
        save_sections(md_name, current_sections, suffix="raw")
        return jsonify({
            "topic": topic,
            "sections": [
                {"header": header, "content": content}
                for header, content in current_sections.items()
            ],
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/save", methods=["POST"])
def save():
    global current_sections
    data = request.get_json()
    responses: dict = data.get("responses", {})       # {header: edited_content}
    topic = data.get("topic") or current_topic

    # Update in-memory sections; textarea value is "header\n\ncontent"
    new_sections = {}
    for old_header, full_text in current_sections.items():
        if old_header in responses:
            raw = responses[old_header]
            first_nl = raw.find("\n")
            if first_nl == -1:
                new_header, new_content = raw.strip(), ""
            else:
                new_header = raw[:first_nl].strip()
                new_content = raw[first_nl:].lstrip("\n")
            new_sections[new_header] = new_content
        else:
            new_sections[old_header] = full_text
    current_sections = new_sections

    # Write back to the original .md file
    md_path = Path("data/md") / Path(current_md_name).name
    md_path.write_text(reconstruct_md(current_sections), encoding="utf-8")

    return jsonify({"message": f"Saved {len(responses)} section(s) to {md_path.name}.", "topic": topic})


@app.route("/api/topic", methods=["GET"])
def get_topic():
    """Return the currently loaded topic (for downstream agents to query)."""
    return jsonify({"topic": current_topic})


if __name__ == "__main__":
    app.run(debug=True)
