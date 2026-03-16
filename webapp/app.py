from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/echo", methods=["POST"])
def echo():
    data = request.get_json()
    message = data.get("message", "")
    return jsonify({"reply": f"You said: {message}"})


if __name__ == "__main__":
    app.run(debug=True)
