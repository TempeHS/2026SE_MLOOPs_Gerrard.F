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
import io
import base64
import matplotlib

matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt

app_log = logging.getLogger(__name__)
logging.basicConfig(
    filename="security_log.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)

app = Flask(__name__)
app.secret_key = b"_53oi3uriq9pifpff;apl"
csrf = CSRFProtect(app)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "multi_linear_model.sav")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "predict"):
        raise TypeError("Loaded object does not have predict().")
    app.logger.critical(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    app.logger.critical(f"Failed to load model from {MODEL_PATH}: {str(e)}")
    model = None

# Feature config — must match training order
FEATURE_NAMES = ["k/d", "matchs_played", "headshot_%", "dmg/rnd"]
TARGET_LABEL = "win%"

# MMR thresholds — adjust to match your data distribution
MMR_THRESHOLDS = {
    "high": {"min": 70, "label": "High MMR", "detail": "Performing above average"},
    "medium": {"min": 45, "label": "Average MMR", "detail": "Performing at average"},
    "low": {"min": 0, "label": "Low MMR", "detail": "Performing below average"},
}


def classify_mmr(predicted_win_pct: float) -> dict:
    """Classify predicted win% into an MMR tier."""
    if predicted_win_pct >= MMR_THRESHOLDS["high"]["min"]:
        return MMR_THRESHOLDS["high"]
    elif predicted_win_pct >= MMR_THRESHOLDS["medium"]["min"]:
        return MMR_THRESHOLDS["medium"]
    else:
        return MMR_THRESHOLDS["low"]


def build_plots(numeric_features: list, predicted_value: float) -> list:
    """Generate one integrated base64 plot with all feature sensitivities."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Common x-axis: percent change from user input
    pct_change = np.linspace(-0.5, 0.5, 200)  # -50% to +50%

    for i, feature_name in enumerate(FEATURE_NAMES):
        X_sweep = np.tile(numeric_features, (len(pct_change), 1)).astype(float)

        base_val = float(numeric_features[i])
        if base_val == 0:
            # If base is zero, use small absolute sweep around zero
            sweep_vals = np.linspace(-1.0, 1.0, len(pct_change))
            X_sweep[:, i] = sweep_vals
        else:
            X_sweep[:, i] = base_val * (1.0 + pct_change)

        y_sweep = model.predict(X_sweep)
        ax.plot(pct_change * 100, y_sweep, linewidth=2, label=feature_name)

    # Mark the user's current prediction at 0% change
    ax.scatter(
        [0], [predicted_value], color="#e24a4a", zorder=6, label="Your prediction"
    )
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax.axhline(predicted_value, color="#2ecc71", linestyle="--", linewidth=1)

    ax.set_title(f"Performance against Player-base ({TARGET_LABEL})", fontsize=12)
    ax.set_xlabel("Features", fontsize=10)
    ax.set_ylabel(TARGET_LABEL, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return [{"feature": "Performance against Player-base", "image": img_b64}]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


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
                        "error": f"Expected {len(FEATURE_NAMES)} features: {FEATURE_NAMES}"
                    }
                ),
                400,
            )

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

        # Predict and clamp to valid domain [0, 100]
        predicted_value = float(prediction[0])
        predicted_value = np.clip(predicted_value, 0, 100)

        mmr_class = classify_mmr(predicted_value)
        plots = build_plots(numeric_features, predicted_value)

        return (
            jsonify(
                {
                    "status": "success",
                    "target": TARGET_LABEL,
                    "predicted_value": predicted_value,
                    "feature_order": FEATURE_NAMES,
                    "mmr_label": mmr_class["label"],
                    "mmr_detail": mmr_class["detail"],
                    "plots": plots,
                }
            ),
            200,
        )

    except Exception as e:
        app.logger.critical(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
