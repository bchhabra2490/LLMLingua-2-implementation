import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from compressor import CompressorService

app = Flask(__name__)
CORS(app)

compressor = CompressorService()


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "checkpoint": compressor.checkpoint_path.name,
        }
    )


@app.post("/api/compress")
def compress_text():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    ratio = payload.get("ratio", 0.5)

    if not text:
        return jsonify({"error": "Prompt text is required."}), 400

    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        return jsonify({"error": "Ratio must be a number between 0.05 and 1."}), 400

    if not 0.05 <= ratio <= 1.0:
        return jsonify({"error": "Ratio must be between 0.05 and 1."}), 400

    result = compressor.run(text, ratio)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
