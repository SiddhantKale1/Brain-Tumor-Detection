from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MriValidationResult:
    is_valid: bool
    reason: str
    mean_intensity: float | None = None
    std_intensity: float | None = None


def validate_brain_mri(
    image_bgr: np.ndarray,
    *,
    min_mean: float = 15.0,
    min_std: float = 25.0,
) -> MriValidationResult:
    """
    Simple heuristic validation:
    - Convert to grayscale
    - Compute mean + standard deviation of pixel intensities
    - If either is too low, reject as unlikely to be a brain MRI
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return MriValidationResult(is_valid=False, reason="Invalid input: unreadable image")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))

    if mean < min_mean or std < min_std:
        return MriValidationResult(
            is_valid=False,
            reason="Invalid input: Not a brain MRI image",
            mean_intensity=mean,
            std_intensity=std,
        )

    return MriValidationResult(
        is_valid=True,
        reason="OK",
        mean_intensity=mean,
        std_intensity=std,
    )


def validate_brain_mri_file(path: str, **kwargs) -> MriValidationResult:
    image = cv2.imread(path)
    return validate_brain_mri(image, **kwargs)

