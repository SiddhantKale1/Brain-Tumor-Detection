from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_FILENAME = "brain_tumor_model.keras"


def compute_class_weights(train_dir: str) -> dict[int, float]:
    """
    Computes balanced class weights inversely proportional to class frequencies
    to prevent majority-class bias and improve minority recall.
    """
    counts = {}
    total = 0
    for idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(train_dir, class_name)
        if os.path.isdir(class_folder):
            c = len([f for f in os.listdir(class_folder) if not f.startswith(".")])
            counts[idx] = c
            total += c
        else:
            counts[idx] = 1

    num_classes = len(CLASS_NAMES)
    weights = {}
    for idx, count in counts.items():
        weights[idx] = float(total / (num_classes * max(count, 1)))

    print("Computed Class Weights:")
    for idx, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:12s}: {counts.get(idx, 0):4d} images -> weight {weights[idx]:.3f}")
    return weights


def build_dataset(data_dir: str, shuffle: bool) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )

    autotune = tf.data.AUTOTUNE
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(1000)
    return dataset.prefetch(buffer_size=autotune)


def build_model(num_classes: int = 4, fine_tune: bool = False) -> tf.keras.Model:
    # Only horizontal flip is anatomically valid for brain MRI (vertical inversion flips cranial/caudal axes)
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.ResNet50(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    if fine_tune:
        base_model.trainable = True
        # Fine-tune deeper residual blocks (conv5)
        fine_tune_at = 140
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=fine_tune)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(
        128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    if fine_tune:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate for fine-tuning
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


def evaluate_detailed_metrics(model: tf.keras.Model, test_ds: tf.data.Dataset) -> None:
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model(images, training=False)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds.numpy(), axis=-1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    except ImportError:
        # Fallback simple per-class accuracy
        print("\nPer-Class Accuracy:")
        for idx, name in enumerate(CLASS_NAMES):
            mask = y_true == idx
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == idx)
                print(f"  {name:12s}: {acc * 100:.2f}% ({np.sum(mask)} samples)")


def main():
    train_dir = os.path.join("dataset", "train")
    val_dir = os.path.join("dataset", "val")
    test_dir = os.path.join("dataset", "test")

    for split_dir, split_name in (
        (train_dir, "train"),
        (val_dir, "val"),
        (test_dir, "test"),
    ):
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected leakage-safe dataset split at '{split_dir}'. "
                f"Run 'python prepare_dataset_splits.py' first to create dataset/{split_name}."
            )

    class_weights = compute_class_weights(train_dir)
    train_ds = build_dataset(train_dir, shuffle=True)
    val_ds = build_dataset(val_dir, shuffle=False)
    test_ds = build_dataset(test_dir, shuffle=False)

    model = build_model(num_classes=len(CLASS_NAMES), fine_tune=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_FILENAME,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    # Initial training with frozen base
    print("\nStage 1: Initial training with frozen base...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Save weights for fine-tuning
    weights_path = "initial_weights.weights.h5"
    model.save_weights(weights_path)
    print(f"Saved initial weights: {weights_path}")

    # Stage 2: Fine-tuning
    print("\nStage 2: Fine-tuning deeper layers...")
    model = build_model(num_classes=len(CLASS_NAMES), fine_tune=True)
    model.load_weights(weights_path)

    fine_tune_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_FILENAME,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-7
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=fine_tune_callbacks,
        class_weight=class_weights,
    )

    # Reload best saved checkpoint for final evaluation
    if os.path.exists(MODEL_FILENAME):
        print(f"\nLoading best checkpoint from '{MODEL_FILENAME}' for evaluation...")
        model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
        model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    evaluate_detailed_metrics(model, test_ds)


if __name__ == "__main__":
    main()
