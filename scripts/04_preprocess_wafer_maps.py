from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class PreprocessingError(RuntimeError):
    """Raised when serving preprocessing fails."""


def resize_and_pad_categorical(
    wafer: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    if wafer.ndim != 2 or wafer.size == 0:
        raise PreprocessingError("Wafer map must be a non-empty 2D array.")

    source_height, source_width = wafer.shape
    scale = min(
        target_height / source_height,
        target_width / source_width,
    )
    resized_height = max(
        1,
        min(target_height, int(round(source_height * scale))),
    )
    resized_width = max(
        1,
        min(target_width, int(round(source_width * scale))),
    )

    resized = cv2.resize(
        wafer,
        dsize=(resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros(
        (target_height, target_width),
        dtype=np.uint8,
    )
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    bottom = top + resized_height
    right = left + resized_width
    canvas[top:bottom, left:right] = resized.astype(np.uint8, copy=False)

    return canvas, {
        "source_height": int(source_height),
        "source_width": int(source_width),
        "resized_height": int(resized_height),
        "resized_width": int(resized_width),
        "padding_top": int(top),
        "padding_bottom": int(target_height - bottom),
        "padding_left": int(left),
        "padding_right": int(target_width - right),
        "scale": float(scale),
    }


def one_hot_encode_wafer(categorical_map: np.ndarray) -> np.ndarray:
    channels = np.stack(
        [
            categorical_map == 0,
            categorical_map == 1,
            categorical_map == 2,
        ],
        axis=0,
    ).astype(np.uint8)

    if not np.all(channels.sum(axis=0) == 1):
        raise PreprocessingError(
            "One-hot tensor must contain one active channel per pixel."
        )
    return channels
