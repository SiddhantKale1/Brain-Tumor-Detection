from __future__ import annotations

import os
import uuid
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf


def _iter_layers_recursive(model: tf.keras.Model):
    for layer in model.layers:
        yield layer
        if isinstance(layer, tf.keras.Model):
            yield from _iter_layers_recursive(layer)


def _find_layer_by_name_recursive(model: tf.keras.Model, name: str) -> tf.keras.layers.Layer:
    for layer in _iter_layers_recursive(model):
        if layer.name == name:
            return layer
    raise ValueError(f"Layer '{name}' not found (searched recursively).")


def _find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    layers = list(_iter_layers_recursive(model))
    for layer in reversed(layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("Could not find a Conv2D layer for Grad-CAM.")


def _find_nested_base_model(model: tf.keras.Model) -> tf.keras.Model:
    nested_models = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and any(isinstance(sub_layer, tf.keras.layers.Conv2D) for sub_layer in layer.layers)
    ]
    if not nested_models:
        raise ValueError("Could not find a nested base model for Grad-CAM.")
    return nested_models[-1]


def _build_gradcam_models(
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str],
) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    base_model = _find_nested_base_model(model)
    base_model_index = model.layers.index(base_model)
    if base_model_index == 0:
        raise ValueError("Base CNN is in an unexpected position for Grad-CAM.")

    if last_conv_layer_name is None:
        last_conv_layer = _find_last_conv_layer(base_model)
    else:
        last_conv_layer = _find_layer_by_name_recursive(base_model, last_conv_layer_name)

    pre_base_tensor = model.layers[base_model_index - 1].output
    preprocess_input_model = tf.keras.Model(model.inputs, pre_base_tensor)

    conv_output_model = tf.keras.Model(
        base_model.input,
        [last_conv_layer.output, base_model.output],
    )

    classifier_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = classifier_input
    passed_base_model = False
    for layer in model.layers:
        if passed_base_model:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
        elif layer is base_model:
            passed_base_model = True

    classifier_model = tf.keras.Model(classifier_input, x)
    return preprocess_input_model, conv_output_model, classifier_model


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    *,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap normalized to [0, 1] with shape (H, W).
    Expects img_array shape (1, H, W, 3).
    If pred_index is None, defaults to the argmax predicted class.
    """
    preprocess_input_model, conv_output_model, classifier_model = _build_gradcam_models(
        model,
        last_conv_layer_name,
    )

    with tf.GradientTape() as tape:
        base_inputs = preprocess_input_model(img_array, training=False)
        conv_outputs, base_outputs = conv_output_model(base_inputs, training=False)
        tape.watch(conv_outputs)
        predictions = classifier_model(base_outputs, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return np.zeros(conv_outputs.shape[1:3], dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy()


def overlay_heatmap_on_bgr(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay


def save_gradcam_overlay(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    *,
    out_dir: str,
    filename: Optional[str] = None,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"gradcam_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(out_dir, filename)

    overlay = overlay_heatmap_on_bgr(image_bgr, heatmap)
    ok = cv2.imwrite(out_path, overlay)
    if not ok:
        raise RuntimeError(f"Failed to write Grad-CAM output image to '{out_path}'.")
    return out_path, filename
