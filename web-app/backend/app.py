import os
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from compressor import CompressorService

STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"

app = Flask(__name__, static_folder=None)
CORS(app)

_compressor = None
_load_error = None
_load_lock = threading.Lock()


def get_compressor() -> CompressorService:
    global _compressor, _load_error

    if _compressor is not None:
        return _compressor

    with _load_lock:
        if _compressor is not None:
            return _compressor
        if _load_error is not None:
            raise RuntimeError(_load_error)

        try:
            _compressor = CompressorService()
            return _compressor
        except Exception as exc:
            _load_error = str(exc)
            raise


def start_model_load():
    def _load():
        try:
            get_compressor()
        except Exception:
            pass

    threading.Thread(target=_load, daemon=True).start()


@app.get("/api/health")
def health():
    payload = {
        "status": "ok",
        "model_loaded": _compressor is not None,
        "model_error": _load_error,
    }
    if _compressor is not None:
        payload["checkpoint"] = _compressor.checkpoint_path.name
    return jsonify(payload)


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

    try:
        result = get_compressor().run(text, ratio)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify(result)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    if path.startswith("api/"):
        return jsonify({"error": "Not found."}), 404

    if STATIC_DIR.exists():
        target = STATIC_DIR / path
        if path and target.is_file():
            return send_from_directory(STATIC_DIR, path)
        return send_from_directory(STATIC_DIR, "index.html")

    return jsonify({"message": "Frontend not built. Run npm run build in web-app/frontend."}), 200


start_model_load()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
