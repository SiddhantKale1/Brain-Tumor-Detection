from __future__ import annotations

import os

import tensorflow as tf


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_FILENAME = "brain_tumor_model.keras"


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
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.2),
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
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
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
    print("Initial training with frozen base...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks,
    )

    # Save weights for fine-tuning
    weights_path = "initial_weights.weights.h5"
    model.save_weights(weights_path)
    print(f"Saved initial weights: {weights_path}")

    # Fine-tuning
    print("Fine-tuning...")
    model = build_model(num_classes=len(CLASS_NAMES), fine_tune=True)
    model.load_weights(weights_path)

    fine_tune_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-7
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,  # More epochs for fine-tuning
        callbacks=fine_tune_callbacks,
    )

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    model.save(MODEL_FILENAME)
    print(f"Saved: {MODEL_FILENAME}")


if __name__ == "__main__":
    main()

