from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np


@dataclass(frozen=True)
class MriValidationResult:
    is_valid: bool
    reason: str
    mean_intensity: float | None = None
    std_intensity: float | None = None
    color_variation: float | None = None
    edge_density: float | None = None


def validate_brain_mri(
    image_bgr: np.ndarray,
    *,
    min_mean: float = 15.0,
    min_std: float = 25.0,
    max_color_variation: float = 25.0,
    min_edge_density: float = 0.01,
    min_dimension: int = 64,
) -> MriValidationResult:
    """
    Validates whether an input image conforms to brain MRI characteristics:
    1. Valid non-empty array and sufficient resolution.
    2. Grayscale/monochrome check: MRI scans should not exhibit large color variations across RGB channels.
    3. Minimum mean intensity: Rejects blank or almost completely black images.
    4. Minimum standard deviation: Rejects flat uniform images.
    5. Structural edge density: Rejects images with insufficient edge structure (e.g. solid blocks).
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return MriValidationResult(is_valid=False, reason="Invalid input: unreadable image")

    h, w = image_bgr.shape[:2]
    if h < min_dimension or w < min_dimension:
        return MriValidationResult(
            is_valid=False,
            reason=f"Invalid image resolution: image is {w}x{h}, minimum required is {min_dimension}x{min_dimension}",
        )

    # 1. Color check (MRI scans are monochrome / grayscale)
    if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
        b, g, r = cv2.split(image_bgr)
        color_variation = float(np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32))) +
                                np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32))))
        if color_variation > max_color_variation:
            return MriValidationResult(
                is_valid=False,
                reason="Invalid input: Image contains significant color. Brain MRI scans must be grayscale.",
                color_variation=color_variation,
            )
    else:
        color_variation = 0.0

    # 2. Intensity & Contrast checks
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    mean = float(np.mean(gray))
    std = float(np.std(gray))

    if mean < min_mean:
        return MriValidationResult(
            is_valid=False,
            reason="Invalid input: Image is too dark or empty to be a valid MRI scan.",
            mean_intensity=mean,
            std_intensity=std,
            color_variation=color_variation,
        )

    if std < min_std:
        return MriValidationResult(
            is_valid=False,
            reason="Invalid input: Image contrast is too low to be a valid MRI scan.",
            mean_intensity=mean,
            std_intensity=std,
            color_variation=color_variation,
        )

    # 3. Structural Edge Check
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)
    if edge_density < min_edge_density:
        return MriValidationResult(
            is_valid=False,
            reason="Invalid input: Insufficient structural anatomical detail detected.",
            mean_intensity=mean,
            std_intensity=std,
            color_variation=color_variation,
            edge_density=edge_density,
        )

    return MriValidationResult(
        is_valid=True,
        reason="OK",
        mean_intensity=mean,
        std_intensity=std,
        color_variation=color_variation,
        edge_density=edge_density,
    )


def validate_brain_mri_file(path: Union[str, Path], **kwargs) -> tuple[MriValidationResult, np.ndarray | None]:
    """
    Reads an image from path and validates it. Returns both the validation result
    and the loaded image array to avoid redundant disk I/O.
    """
    image = cv2.imread(str(path))
    result = validate_brain_mri(image, **kwargs)
    return result, image
