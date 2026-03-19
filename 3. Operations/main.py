from flask import Flask
from flask import redirect
from flask import render_template
from flask import request, session, url_for
from flask import jsonify
import requests
from flask_wtf import CSRFProtect
from flask_csp.csp import csp_header
import os
import logging
import bcrypt
import pickle
import numpy as np

# Code snippet for logging a message
# app.logger.critical("message")

app_log = logging.getLogger(__name__)
logging.basicConfig(
    filename="security_log.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)

# Generate a unique basic 16 key: https://acte.ltd/utils/randomkeygen
app = Flask(__name__)
app.secret_key = b"_53oi3uriq9pifpff;apl"
csrf = CSRFProtect(app)

# Load your model (os-based, robust path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "polynomial_model.sav")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "predict"):
        raise TypeError("Loaded object does not have predict().")

    app.logger.critical(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    app.logger.critical(f"Failed to load model from {MODEL_PATH}: {str(e)}")
    model = None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ...existing code...

# Keep this order exactly the same as training FEATURE_COLS
FEATURE_NAMES = ["k/d", "matchs_played"]
TARGET_LABEL = "win%"


@app.route("/api/predict", methods=["POST"])
@csrf.exempt
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "No JSON data provided"}), 400

        features = data.get("features", [])
        if not isinstance(features, list) or len(features) != len(FEATURE_NAMES):
            return (
                jsonify(
                    {
                        "error": f"Expected {len(FEATURE_NAMES)} features in this order: {FEATURE_NAMES}"
                    }
                ),
                400,
            )

        # Strict numeric parsing
        try:
            numeric_features = [float(v) for v in features]
        except (TypeError, ValueError):
            return jsonify({"error": "All features must be numeric"}), 400

        X = np.array(numeric_features, dtype=float).reshape(1, -1)

        try:
            prediction = model.predict(X)
        except ValueError as e:
            return (
                jsonify(
                    {"error": "Model input shape/type mismatch", "details": str(e)}
                ),
                400,
            )

        return (
            jsonify(
                {
                    "status": "success",
                    "target": TARGET_LABEL,
                    "predicted_value": float(prediction[0]),
                    "feature_order": FEATURE_NAMES,
                }
            ),
            200,
        )

    except Exception as e:
        app.logger.critical(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
