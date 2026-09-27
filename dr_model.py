"""
dr_model.py — Diabetic Retinopathy ML Pipeline
Covers: preprocessing, feature extraction, model building, training, evaluation
"""

import base64
import io
import os
import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────── Constants ───────────────────────────
IMG_SIZE = 224  # Input size expected by the CNN
NUM_CLASSES = 5  # 0-No DR, 1-Mild, 2-Moderate, 3-Severe, 4-Proliferative
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# Global cache for feature extraction model
_feature_model = None


# ─────────────────────────── Preprocessing ───────────────────────
def load_image(image_path: str) -> np.ndarray:
    """Load an image and convert to BGR numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image at: {image_path}")
    return img


def preprocess_image(image_path: str):
    """Full preprocessing pipeline:

      1. Resize to IMG_SIZE × IMG_SIZE
      2. Convert to LAB colour space
      3. Apply CLAHE to L-channel (contrast enhancement)
      4. Gaussian blur to reduce noise
      5. Normalise to [0, 1]
    Returns:
        original_b64  : base64-encoded JPEG of the original image
        processed_b64 : base64-encoded JPEG of the processed image
        stats         : dict with image statistics
    """
    img = load_image(image_path)
    original_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # ── CLAHE ──────────────────────────────────────────────────────
    lab = cv2.cvtColor(original_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # ── Gaussian blur ──────────────────────────────────────────────
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    original_b64 = _ndarray_to_b64(original_resized)
    processed_b64 = _ndarray_to_b64(blurred)

    stats = {
        "original_mean": float(np.mean(original_resized)),
        "processed_mean": float(np.mean(blurred)),
        "original_std": float(np.std(original_resized)),
        "processed_std": float(np.std(blurred)),
        "width": original_resized.shape[1],
        "height": original_resized.shape[0],
    }
    return original_b64, processed_b64, stats


# ─────────────────────────── Feature Extraction ──────────────────
def extract_features(image_path: str):
    """Extract features using a MobileNetV2 backbone (transfer learning).

    Returns:
        feature_maps_b64 : 9 feature-map thumbnails as base64 PNGs
        feature_shape    : shape of the extracted feature tensor
    """
    global _feature_model
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = load_image(image_path)
    img_rgb = cv2.cvtColor(
        cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB
    )
    x = preprocess_input(np.expand_dims(img_rgb.astype("float32"), axis=0))

    # Lazy load & cache feature model
    if _feature_model is None:
        try:
            base_model = MobileNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
            )
            _feature_model = tf.keras.Model(
                inputs=base_model.input,
                outputs=base_model.get_layer("block_3_expand_relu").output,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MobileNetV2 weights. Please delete corrupted file in C:\\Users\\<user>\\.keras\\models\\ and restart. Error: {str(e)}"
            )

    feature_maps = _feature_model.predict(x, verbose=0)[0]  # (H, W, C)

    # Pick 9 channels to visualise
    n = min(9, feature_maps.shape[-1])
    thumbnails = []
    for i in range(n):
        fm = feature_maps[:, :, i]
        fm_range=fm.max()-fm.min()
        fm_norm = ((fm - fm.min()) / (fm_range + 1e-8) * 255).astype("uint8")
        colored = cv2.applyColorMap(fm_norm, cv2.COLORMAP_VIRIDIS)
        thumbnails.append(_ndarray_to_b64(colored))

    return thumbnails, list(feature_maps.shape)


# ─────────────────────────── Model Building ──────────────────────
def build_cnn_model():
    """Build a lightweight CNN for 5-class DR classification.

    Architecture: Conv(32) → Pool → Conv(64) → Pool → Conv(128) → Pool
                  → Flatten → Dense(256) → Dropout → Dense(5, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="DR_CNN",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────── Training ────────────────────────────
def train_model(epochs: int = 10, batch_size: int = 16):
    """Demo training using synthetically generated data.

    In a real project, replace X_train/X_test/y_train/y_test
    with your actual dataset (e.g., from the APTOS 2019 Kaggle dataset).

    Returns:
        history : dict with keys 'accuracy', 'val_accuracy', 'loss', 'val_loss'
        model   : trained Keras model
        X_test  : test images (numpy)
        y_test  : ground-truth labels (numpy)
    """
    import tensorflow as tf

    np.random.seed(42)
    tf.random.set_seed(42)

    # ── Synthetic dataset (200 train, 50 test) ─────────────────────
    N_TRAIN, N_TEST = 200, 50
    X_train = np.random.rand(N_TRAIN, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    y_train = np.random.randint(0, NUM_CLASSES, N_TRAIN)
    X_test = np.random.rand(N_TEST, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    y_test = np.random.randint(0, NUM_CLASSES, N_TEST)

    model = build_cnn_model()
    hist = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    history = {
        "accuracy": [round(v, 4) for v in hist.history["accuracy"]],
        "val_accuracy": [round(v, 4) for v in hist.history["val_accuracy"]],
        "loss": [round(v, 4) for v in hist.history["loss"]],
        "val_loss": [round(v, 4) for v in hist.history["val_loss"]],
    }
    return history, model, X_test, y_test


# ─────────────────────────── Evaluation ──────────────────────────
def evaluate_model(model, X_test, y_test):
    """Compute classification metrics and generate:

      • Confusion matrix  (base64 PNG)
      • ROC curve         (base64 PNG — macro-average OvR)

    Returns dict with:
        accuracy, precision, recall, f1,
        confusion_matrix (list[list[int]]),
        class_names,
        roc_auc (float, macro),
        confusion_matrix_b64, roc_curve_b64
    """
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

    # ── Confusion Matrix Plot ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", color="white")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES, color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True", color="white")
    ax.set_title("Confusion Matrix", color="white")
    ax.tick_params(colors="white")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            str(val),
            ha="center",
            va="center",
            color="white" if val < cm.max() / 2 else "black",
            fontsize=9,
        )
    plt.tight_layout()
    cm_b64 = _fig_to_b64(fig)
    plt.close(fig)

    # ── ROC Curve (macro OvR) ──────────────────────────────────────
    y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
    fpr_all, tpr_all, roc_auc_all = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr_all[i], tpr_all[i], _ = roc_curve(
            y_test_bin[:, i], y_pred_prob[:, i]
        )
        roc_auc_all[i] = auc(fpr_all[i], tpr_all[i])

    colors = ["#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8"]
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.set_facecolor("#1a1a2e")
    for i, (col, name) in enumerate(zip(colors, CLASS_NAMES)):
        ax2.plot(
            fpr_all[i],
            tpr_all[i],
            color=col,
            lw=1.8,
            label=f"{name} (AUC={roc_auc_all[i]:.2f})",
        )
    ax2.plot([0, 1], [0, 1], "w--", lw=1)
    ax2.set_xlabel("False Positive Rate", color="white")
    ax2.set_ylabel("True Positive Rate", color="white")
    ax2.set_title("ROC Curve (One-vs-Rest)", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#0d0d1a", labelcolor="white", fontsize=8)
    plt.tight_layout()
    roc_b64 = _fig_to_b64(fig2)
    plt.close(fig2)

    macro_auc = float(np.mean(list(roc_auc_all.values())))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(macro_auc, 4),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "confusion_matrix_b64": cm_b64,
        "roc_curve_b64": roc_b64,
    }
# ─────────────────────────── Prediction ─────────────────────────
def predict_image(model, image_path: str):
    """Predict diabetic retinopathy severity for one retinal image."""

    img = load_image(image_path)

    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    probabilities = model.predict(img, verbose=0)[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return {
        "class_id": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(NUM_CLASSES)
        },
    }


# ─────────────────────────── Helpers ─────────────────────────────
def _ndarray_to_b64(img_bgr: np.ndarray) -> str:
    """Encode a BGR numpy image to a base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buf).decode("utf-8")


def _fig_to_b64(fig) -> str:
    """Encode a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
