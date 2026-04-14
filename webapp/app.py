import json
import queue
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
import werkzeug

sys.path.insert(0, str(Path(__file__).parent.parent))
from modular_seg import reconstruct_md, save_sections, segment_md
from doc_preprocess import doc_preprocess

app = Flask(__name__)

# ── In-memory state ──────────────────────────────────────────────────────────
current_md_name = None
current_sections = {}
current_topic = None

VALID_TOPICS    = {"Machine Learning", "Deep Learning", "Generative Models", "Transfer Learning", "Computer Vision", "NLP", "AI for Science", "Others"}
VALID_REVIEWERS = {"reviewer_a", "reviewer_b"}

# job_id -> queue.Queue  (progress events)
_jobs: dict[str, queue.Queue] = {}
# job_id -> structured results dict (populated when job finishes)
_results: dict[str, dict] = {}


# ── Page ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── File upload ──────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_file():
    global current_md_name, current_sections

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    filename = werkzeug.utils.secure_filename(f.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".md"):
        return jsonify({"error": "Only .pdf and .md files are supported."}), 400

    try:
        if ext == ".pdf":
            pdf_dir = Path("data/pdf")
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = pdf_dir / filename
            f.save(str(pdf_path))

            # Convert PDF → MD (may take a while)
            doc_preprocess(filename, pdf_path=str(pdf_dir), md_path="data/md")
            md_filename = Path(filename).with_suffix(".md").name
        else:
            md_dir = Path("data/md")
            md_dir.mkdir(parents=True, exist_ok=True)
            md_path = md_dir / filename
            f.save(str(md_path))
            md_filename = filename

        # Segment the resulting MD file
        current_sections = segment_md(md_filename, md_path="data/md")
        current_md_name  = md_filename
        save_sections(md_filename, current_sections, suffix="raw")

        md_content = (Path("data/md") / md_filename).read_text(encoding="utf-8")

        return jsonify({
            "md_name": md_filename,
            "md_content": md_content,
            "sections": [
                {"header": header, "content": content}
                for header, content in current_sections.items()
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Paper loading & saving ───────────────────────────────────────────────────

@app.route("/api/segments", methods=["POST"])
def segments():
    global current_md_name, current_sections
    data    = request.get_json()
    md_name = data.get("md_name", "")
    md_path = data.get("md_path", "data/md")

    try:
        current_sections = segment_md(md_name, md_path=md_path)
        current_md_name  = md_name
        save_sections(md_name, current_sections, suffix="raw")
        return jsonify({
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
    data      = request.get_json()
    responses = data.get("responses", {})

    # Rebuild sections; each textarea value is "#### Header\n\ncontent…"
    new_sections = {}
    for old_header, full_text in current_sections.items():
        if old_header in responses:
            raw      = responses[old_header]
            first_nl = raw.find("\n")
            if first_nl == -1:
                new_header, new_content = raw.strip(), ""
            else:
                new_header  = raw[:first_nl].strip()
                new_content = raw[first_nl:].lstrip("\n")
            new_sections[new_header] = new_content
        else:
            new_sections[old_header] = full_text
    current_sections = new_sections

    md_path = Path("data/md") / Path(current_md_name).name
    md_path.write_text(reconstruct_md(current_sections), encoding="utf-8")
    return jsonify({"message": f"Saved {len(responses)} section(s) to {md_path.name}."})


# ── Review run ───────────────────────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def run_review():
    global current_topic
    data           = request.get_json()
    topic          = data.get("topic", "")
    reviewer_types = data.get("reviewers", ["reviewer_a", "reviewer_b"])
    n_iter         = max(1, int(data.get("n_iter", 3)))
    api_key        = data.get("api_key", "").strip()

    if topic not in VALID_TOPICS:
        return jsonify({"error": f"Invalid topic. Choose from: {sorted(VALID_TOPICS)}"}), 400
    invalid = [r for r in reviewer_types if r not in VALID_REVIEWERS]
    if invalid:
        return jsonify({"error": f"Unknown reviewer(s): {invalid}"}), 400
    if not reviewer_types:
        return jsonify({"error": "Select at least one reviewer."}), 400
    if not current_md_name:
        return jsonify({"error": "No paper loaded. Go back and load a paper first."}), 400
    if not api_key:
        return jsonify({"error": "API key is required."}), 400

    md_path      = Path("data/md") / Path(current_md_name).name
    paper        = md_path.read_text(encoding="utf-8")
    current_topic = topic

    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    _jobs[job_id]  = q

    def run():
        try:
            from mas_loop import main as mas_main
            result = mas_main(
                paper=paper, topic=topic, n_iter=n_iter,
                reviewer_types=reviewer_types, api_key=api_key,
                on_event=lambda msg: q.put(("status", msg)),
            )
            _results[job_id] = result

            # Save results to results/ folder
            project_root = Path(__file__).parent.parent
            results_dir = project_root / "results"
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%y%m%d%H%M")
            paper_name = Path(current_md_name).stem
            fname = f"{timestamp}_nagent={len(reviewer_types)}_niter={n_iter}_paper={paper_name}.txt"
            (results_dir / fname).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

            q.put(("done", "Review complete!"))
        except Exception as exc:
            q.put(("error", str(exc)))

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


# ── SSE stream ───────────────────────────────────────────────────────────────

@app.route("/api/stream/<job_id>")
def stream(job_id):
    q = _jobs.get(job_id)
    if not q:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        while True:
            try:
                event_type, message = q.get(timeout=120)
                yield f"data: {json.dumps({'type': event_type, 'message': message})}\n\n"
                if event_type in ("done", "error"):
                    _jobs.pop(job_id, None)
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat', 'message': ''})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Results ──────────────────────────────────────────────────────────────────

@app.route("/api/results/<job_id>")
def get_results(job_id):
    result = _results.get(job_id)
    if result is None:
        return jsonify({"error": "Results not ready or job not found."}), 404
    return jsonify(result)


@app.route("/api/topic", methods=["GET"])
def get_topic():
    return jsonify({"topic": current_topic})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
