from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from gradcam import make_gradcam_heatmap, overlay_heatmap_on_bgr


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "brain_tumor_model.keras"
OUTPUT_DIR = Path("debug_outputs")


def load_model_for_debugging(model_path: str = MODEL_PATH) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Run main.py first.")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(
            model_path, custom_objects={"loss_fn": lambda y, p: p}, compile=False
        )
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_class_distribution(dataset_dir: str = "dataset") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    data = {s: [] for s in splits}

    for s in splits:
        split_path = Path(dataset_dir) / s
        for c in CLASS_NAMES:
            c_path = split_path / c
            count = len(list(c_path.glob("*.*"))) if c_path.exists() else 0
            data[s].append(count)

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(splits):
        ax.bar(x + (i - 1) * width, data[s], width, label=s.capitalize())

    ax.set_ylabel("Number of Images")
    ax.set_title("Dataset Class Distribution Across Splits")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    plt.tight_layout()
    out_file = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved class distribution plot to: {out_file}")


def evaluate_and_plot_confusion_matrix(model: tf.keras.Model, test_dir: str = "dataset/test") -> None:
    if not os.path.isdir(test_dir):
        print(f"Test directory '{test_dir}' not found. Skipping evaluation.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    y_true = []
    y_pred = []
    misclassified_samples = []

    for images, labels in test_ds:
        preds = model(images, training=False).numpy()
        pred_labels = np.argmax(preds, axis=-1)
        true_labels = labels.numpy()

        for img, true_lbl, pred_lbl, prob in zip(images.numpy(), true_labels, pred_labels, preds):
            if true_lbl != pred_lbl and len(misclassified_samples) < 5:
                misclassified_samples.append((img, true_lbl, pred_lbl, prob[pred_lbl]))

        y_true.extend(true_labels)
        y_pred.extend(pred_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n--- Model Evaluation Report ---")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
        fig.colorbar(cax)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center", size="large")

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="left")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix", pad=20)
        plt.tight_layout()
        cm_out = OUTPUT_DIR / "confusion_matrix.png"
        plt.savefig(cm_out, dpi=150)
        plt.close()
        print(f"Saved confusion matrix plot to: {cm_out}")
    except ImportError:
        print("Note: Install 'scikit-learn' for detailed classification report and confusion matrix plotting.")


def generate_sample_gradcam(model: tf.keras.Model, sample_image_path: str) -> None:
    if not os.path.exists(sample_image_path):
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(sample_image_path)
    if img_bgr is None:
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)

    preds = model(img_array, training=False).numpy()[0]
    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_NAMES[pred_idx]

    heatmap = make_gradcam_heatmap(img_array, model, pred_index=pred_idx)
    overlay = overlay_heatmap_on_bgr(img_bgr, heatmap)

    out_file = OUTPUT_DIR / f"debug_gradcam_{pred_label}.jpg"
    cv2.imwrite(str(out_file), overlay)
    print(f"Generated sample Grad-CAM debug overlay: {out_file}")


def main():
    print("Running model diagnostics and debug checks...")
    plot_class_distribution()

    if os.path.exists(MODEL_PATH):
        model = load_model_for_debugging(MODEL_PATH)
        evaluate_and_plot_confusion_matrix(model)
    else:
        print(f"Model checkpoint '{MODEL_PATH}' not found. Run 'python main.py' to train first.")


if __name__ == "__main__":
    main()
