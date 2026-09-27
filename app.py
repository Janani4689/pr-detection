"""
app.py — Flask REST API for Diabetic Retinopathy Web Application
"""

import os
import uuid
import cv2
import dr_model as drm
import numpy as np

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

STATE_FILE = os.path.join(UPLOAD_FOLDER, "latest_path.txt")

_state = {}


def _allowed(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT
    )


def set_image_path(path):
    with open(STATE_FILE, "w") as f:
        f.write(path)


def get_image_path():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return f.read().strip()
    return None


# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]

    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed(f.filename):
        return jsonify({"error": "Invalid image format"}), 400

    ext = f.filename.rsplit(".", 1)[1].lower()

    filename = f"{uuid.uuid4().hex}.{ext}"

    save_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    f.save(save_path)

    set_image_path(save_path)

    # Reset previous model state
    _state.pop("model", None)
    _state.pop("X_test", None)
    _state.pop("y_test", None)

    try:

        original_b64, _, _ = drm.preprocess_image(save_path)

        return jsonify({
            "message": "Image uploaded successfully",
            "filename": filename,
            "preview_b64": original_b64,
            "file_size_kb": round(
                os.path.getsize(save_path) / 1024,
                2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────

@app.route("/api/preprocess", methods=["POST"])
def preprocess():

    image_path = get_image_path()

    if not image_path or not os.path.exists(image_path):
        return jsonify({
            "error": "No image uploaded yet"
        }), 400

    try:

        orig_b64, proc_b64, stats = drm.preprocess_image(
            image_path
        )

        return jsonify({
            "original_b64": orig_b64,
            "processed_b64": proc_b64,
            "stats": stats
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

@app.route("/api/extract", methods=["POST"])
def extract():

    image_path = get_image_path()

    if not image_path or not os.path.exists(image_path):
        return jsonify({
            "error": "No image uploaded yet"
        }), 400

    try:

        thumbnails, shape = drm.extract_features(
            image_path
        )

        return jsonify({

            "feature_maps": thumbnails,

            "feature_shape": shape,

            "description":
                f"MobileNetV2 feature output shape: "
                f"{shape[0]} × {shape[1]} × {shape[2]}. "
                f"Showing 9 representative feature maps."

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def train():
    

    body = request.get_json(silent=True) or {}

    try:
        epochs = int(body.get("epochs", 10))
    except:
        epochs = 10

    try:

        history, model, X_test, y_test = drm.train_model(
            epochs=epochs
        )
        

        _state["model"] = model
        _state["X_test"] = X_test
        _state["y_test"] = y_test

        return jsonify({

            "history": history,

            "epochs": epochs,

            "message":
                f"Training complete — {epochs} epochs."

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────
# PREDICTION ⭐
# ─────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():

    # Check trained model
    if "model" not in _state:

        return jsonify({
            "error": "Please train the model first."
        }), 400

    image_path = get_image_path()

    # Check uploaded image
    if not image_path or not os.path.exists(image_path):

        return jsonify({
            "error": "Please upload a retinal image first."
        }), 400

    try:

        model = _state["model"]

        # Load image
        img = drm.load_image(image_path)

        # Resize
        img = cv2.resize(
            img,
            (drm.IMG_SIZE, drm.IMG_SIZE)
        )

        # BGR → RGB
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # Normalize
        img = img.astype("float32") / 255.0

        # Add batch dimension
        img = np.expand_dims(
            img,
            axis=0
        )

        # Prediction
        probabilities = model.predict(
            img,
            verbose=0
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        class_name = drm.CLASS_NAMES[predicted_class
            
        ]

        probability_data = {}

        for i, name in enumerate(drm.CLASS_NAMES):

            probability_data[name] = round(
                float(probabilities[i]) * 100,
                2
            )

        return jsonify({

            "prediction": class_name,

            "class_index": predicted_class,

            "confidence": round(
                confidence * 100,
                2
            ),

            "probabilities":
                probability_data,

            "message":
                "Prediction generated successfully."

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

@app.route("/api/evaluate", methods=["POST"])
def evaluate():

    if "model" not in _state:

        return jsonify({
            "error": "Train the model first"
        }), 400

    try:

        results = drm.evaluate_model(

            _state["model"],

            _state["X_test"],

            _state["y_test"]

        )

        return jsonify(results)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  Diabetic Retinopathy ML Server")
    print("  Open http://localhost:5000")
    print("=" * 60)

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )