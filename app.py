from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from clinical_engine import DicomHandler, ClinicalAnalyzer, DiagnosticReportGenerator
from gradcam import make_gradcam_heatmap, save_gradcam_overlay
from mri_validator import validate_brain_mri_file


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
REPORT_DIR = STATIC_DIR / "reports"
DEMO_DIR = BASE_DIR / "sample_dicoms"
MODEL_PATH = BASE_DIR / "brain_tumor_model.keras"

CONFIDENCE_THRESHOLD = 0.60
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".dcm", ".dicom")
MAX_FILE_SIZE_MB = 32

DEMO_MAP = {
    "case_01_glioma": DEMO_DIR / "glioma_high_grade.dcm",
    "case_02_meningioma": DEMO_DIR / "meningioma_parasagittal.dcm",
    "case_03_pituitary": DEMO_DIR / "pituitary_sellar_adenoma.dcm",
    "case_04_normal": DEMO_DIR / "normal_control_brain.dcm",
}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Train first: python main.py"
        )

    clinical_analyzer = ClinicalAnalyzer(str(MODEL_PATH))
    model = clinical_analyzer.model

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # In-memory mapping of report_id to file path
    report_cache: dict[str, Path] = {}

    def predict_image_bgr(image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        img = image_rgb.astype(np.float32)
        img_array = np.expand_dims(img, axis=0)

        preds = model(img_array, training=False).numpy()[0]
        probs = np.array(preds, dtype=np.float32)
        idx = int(np.argmax(probs))

        return CLASS_NAMES[idx], float(probs[idx]), probs, img_array, idx

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

    @app.get("/workstation")
    def workstation():
        return render_template("clinical_workstation.html")

    @app.get("/detect")
    def detect():
        return render_template("detect.html")

    @app.route("/api/analyze", methods=["GET", "POST"])
    def api_analyze():
        try:
            demo_case = request.args.get("demo", "").strip()
            target_path = None

            if demo_case and demo_case in DEMO_MAP:
                target_path = DEMO_MAP[demo_case]
                if not target_path.exists():
                    return jsonify({"ok": False, "error": f"Demo scan {demo_case} not found on disk."}), 404
            elif "file" in request.files:
                file = request.files["file"]
                if not file or file.filename == "":
                    return jsonify({"ok": False, "error": "No file uploaded."}), 400

                filename = secure_filename(file.filename)
                if not filename.lower().endswith(ALLOWED_EXTENSIONS):
                    return jsonify({"ok": False, "error": "Unsupported file format. Please upload .dcm, .dicom, .jpg, or .png"}), 400

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                saved_name = f"{stamp}_{filename}"
                target_path = UPLOAD_DIR / saved_name
                file.save(str(target_path))
            else:
                return jsonify({"ok": False, "error": "Missing input: provide 'demo' query or upload 'file'."}), 400

            # 1. Parse Image / DICOM
            image_bgr, metadata = DicomHandler.parse_file(target_path)

            # 2. Run Clinical End-to-End Analysis
            metrics, visuals = clinical_analyzer.analyze(image_bgr, metadata)

            # 3. Generate Structured Diagnostic PDF Report
            report_id = uuid.uuid4().hex[:12]
            report_path = REPORT_DIR / f"report_{report_id}.pdf"

            # Decode raw / segmentation images for PDF embedding
            DiagnosticReportGenerator.generate_pdf_report(
                report_path,
                metadata,
                metrics,
                image_bgr,
                image_bgr,
            )
            report_cache[report_id] = report_path

            return jsonify({
                "ok": True,
                "report_id": report_id,
                "metrics": metrics.to_dict(),
                "metadata": metadata.to_dict(),
                "visuals": visuals,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/api/export-report")
    def export_report():
        report_id = request.args.get("id", "").strip()
        if not report_id or report_id not in report_cache:
            # Fallback to any recent report if ID expired
            recent_reports = list(REPORT_DIR.glob("*.pdf"))
            if recent_reports:
                report_file = recent_reports[-1]
            else:
                return jsonify({"ok": False, "error": "Report not found or expired."}), 404
        else:
            report_file = report_cache[report_id]

        if not report_file.exists():
            return jsonify({"ok": False, "error": "Report file missing from disk."}), 404

        return send_file(
            str(report_file),
            as_attachment=True,
            download_name=f"NeuroScan_Diagnostic_Report_{report_id}.pdf",
            mimetype="application/pdf",
        )

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

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        saved_name = f"{stamp}_{filename}"
        upload_path = UPLOAD_DIR / saved_name
        file.save(str(upload_path))

        # Comprehensive MRI validation (grayscale, contrast, edges, dimensions)
        validation, image_bgr = validate_brain_mri_file(upload_path)
        if not validation.is_valid:
            return render_template(
                "detect.html",
                error=validation.reason,
                uploaded_image=f"uploads/{saved_name}",
            )

        if image_bgr is None:
            return render_template(
                "detect.html",
                error="Invalid input: unreadable image",
                uploaded_image=f"uploads/{saved_name}",
            )

        label, confidence, probs, _, class_idx = predict_image_bgr(image_bgr)

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
            pred_index=class_idx,
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

        pred_idx_param = request.args.get("pred_index", None)
        pred_index = (
            int(pred_idx_param)
            if pred_idx_param is not None and pred_idx_param.isdigit()
            else None
        )

        try:
            upload_path, image_bgr = load_uploaded_image(filename)
            gradcam_filename = f"gradcam_{upload_path.stem}.jpg"
            gradcam_path = UPLOAD_DIR / gradcam_filename

            if not gradcam_path.exists():
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                img_array = np.expand_dims(image_rgb.astype(np.float32), axis=0)

                heatmap = make_gradcam_heatmap(img_array, model, pred_index=pred_index)
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
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "Uploaded image not found."}), 404
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
