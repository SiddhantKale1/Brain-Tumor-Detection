from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from gradcam import make_gradcam_heatmap, save_gradcam_overlay
from mri_validator import validate_brain_mri_file


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)

BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
MODEL_PATH = BASE_DIR / "brain_tumor_model.keras"

CONFIDENCE_THRESHOLD = 0.60
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def create_app() -> Flask:
    app = Flask(__name__)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Train first: python main.py"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"loss_fn": lambda y, p: p},
        compile=False,
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    def predict_image_bgr(image_bgr: np.ndarray):

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        img = image_rgb.astype(np.float32)
        img_array = np.expand_dims(img, axis=0)

        probs = model.predict(img_array, verbose=0)[0]

        # ================= 🔥 PITUITARY FIX =================
        pituitary_prob = probs[3]
        notumor_prob   = probs[2]

        if pituitary_prob > 0.20:
            idx = 3
        elif notumor_prob > 0.85 and pituitary_prob < 0.15:
            idx = 2
        else:
            idx = int(np.argmax(probs))
        # ===================================================

        return CLASS_NAMES[idx], float(probs[idx]), probs, img_array, image_bgr

    def load_uploaded_image(filename: str) -> tuple[Path, np.ndarray]:
        safe_name = os.path.basename(filename)
        upload_path = UPLOAD_DIR / safe_name
        if not upload_path.exists():
            raise FileNotFoundError("Uploaded image was not found.")

        image_bgr = cv2.imread(str(upload_path))
        if image_bgr is None:
            raise ValueError("Uploaded image could not be read.")
        return upload_path, image_bgr

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/detect")
    def detect():
        return render_template("detect.html")

    @app.post("/predict")
    def predict():
        if "file" not in request.files:
            return render_template("detect.html", error="No file part in request.")

        file = request.files["file"]
        if not file or file.filename == "":
            return render_template("detect.html", error="Please choose an image to upload.")

        filename = secure_filename(file.filename)
        if not filename.lower().endswith(ALLOWED_EXTENSIONS):
            return render_template("detect.html", error="Unsupported file type.")

        stamp      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        saved_name = f"{stamp}_{filename}"
        upload_path = UPLOAD_DIR / saved_name
        file.save(str(upload_path))

        # ================= ORIGINAL VALIDATION =================
        validation = validate_brain_mri_file(str(upload_path))
        if not validation.is_valid:
            return render_template(
                "detect.html",
                error=validation.reason,
                uploaded_image=f"uploads/{saved_name}",
            )

        # ================= EXTRA MRI VALIDATION =================
        image_bgr = cv2.imread(str(upload_path))

        if image_bgr is not None:

            # 1. Color check (MRI should be grayscale)
            b, g, r = cv2.split(image_bgr)
            color_variation = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))

            if color_variation > 25:
                return render_template(
                    "detect.html",
                    error="Invalid input: Please upload a brain MRI scan only.",
                    uploaded_image=f"uploads/{saved_name}",
                )

            # 2. Edge structure check
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / edges.size

            if edge_density < 0.01:
                return render_template(
                    "detect.html",
                    error="Invalid MRI scan: insufficient structural detail.",
                    uploaded_image=f"uploads/{saved_name}",
                )
        # =======================================================

        # ================= ORIGINAL FLOW CONTINUES =================
        if image_bgr is None:
            return render_template(
                "detect.html",
                error="Invalid input: unreadable image",
                uploaded_image=f"uploads/{saved_name}",
            )

        label, confidence, probs, img_array, image_bgr = predict_image_bgr(image_bgr)

        tumor_detected = label != "notumor"
        headline = f"Tumor Detected: {label.capitalize()}" if tumor_detected else "No Tumor Detected"

        low_confidence = confidence < CONFIDENCE_THRESHOLD
        confidence_warning = (
            "⚠️ Low confidence prediction — result may be unreliable. "
            "Please consult a specialist." if low_confidence else None
        )

        return render_template(
            "results.html",
            uploaded_image=f"uploads/{saved_name}",
            gradcam_source=saved_name,
            headline=headline,
            tumor_type=label,
            confidence=f"{confidence * 100:.2f}%",
            confidence_warning=confidence_warning,
            low_confidence=low_confidence,
            probs={CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        )

    @app.get("/gradcam")
    def gradcam():
        filename = request.args.get("image", "").strip()
        if not filename:
            return jsonify({"ok": False, "error": "Missing image parameter."}), 400

        try:
            upload_path, image_bgr = load_uploaded_image(filename)
            _, _, _, img_array, image_bgr = predict_image_bgr(image_bgr)
            gradcam_filename = f"gradcam_{upload_path.stem}.jpg"
            gradcam_path = UPLOAD_DIR / gradcam_filename

            if not gradcam_path.exists():
                heatmap = make_gradcam_heatmap(img_array, model)
                save_gradcam_overlay(
                    image_bgr,
                    heatmap,
                    out_dir=str(UPLOAD_DIR),
                    filename=gradcam_filename,
                )

            return jsonify({
                "ok": True,
                "image_url": f"/static/uploads/{gradcam_filename}",
            })
        except Exception as e:
            print(f"[Grad-CAM] Failed: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
