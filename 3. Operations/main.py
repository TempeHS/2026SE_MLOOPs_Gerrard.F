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

# Load your model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    app.logger.critical("Model loaded successfully")
except Exception as e:
    app.logger.critical(f"Failed to load model: {str(e)}")
    model = None


# ...existing code...


# API endpoint for model predictions
@app.route("/api/predict", methods=["POST"])
@csrf.exempt
def predict():
    """
    API endpoint to get predictions from your model
    Expects JSON input with features
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract features - adjust based on your model's input
        features = data.get("features", [])

        if not features:
            return jsonify({"error": "Missing 'features' in request"}), 400

        # Convert to numpy array and reshape if needed
        X = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(X)

        # Log the prediction
        app.logger.critical(f"Prediction made: {prediction}")

        return jsonify({"prediction": prediction.tolist(), "status": "success"}), 200

    except Exception as e:
        app.logger.critical(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Test endpoint
@app.route("/api/health", methods=["GET"])
def health():
    """Check if API and model are running"""
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200


# ...existing code...


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
