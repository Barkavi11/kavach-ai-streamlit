from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class OcclusionResult:
    original_probability: float
    importance_map: np.ndarray
    signed_probability_change: np.ndarray
    evaluated_windows: int


def active_wafer_mask(input_tensor: Tensor) -> Tensor:
    return torch.argmax(input_tensor, dim=0) != 0


def apply_good_die_occlusion(
    image: Tensor,
    *,
    top: int,
    left: int,
    patch_size: int,
    good_die_channel: int = 1,
) -> Tensor:
    perturbed = image.clone()
    height = image.shape[-2]
    width = image.shape[-1]
    bottom = min(top + patch_size, height)
    right = min(left + patch_size, width)
    patch = perturbed[:, top:bottom, left:right]
    active_mask = torch.argmax(patch, dim=0) != 0
    if not bool(active_mask.any()):
        return perturbed
    patch[:, active_mask] = 0.0
    patch[good_die_channel, active_mask] = 1.0
    return perturbed


def sliding_positions(
    *,
    length: int,
    patch_size: int,
    stride: int,
) -> list[int]:
    if patch_size >= length:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    final_position = length - patch_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def calculate_occlusion_sensitivity(
    *,
    model: nn.Module,
    image: Tensor,
    target_class_id: int,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> OcclusionResult:
    with torch.no_grad():
        original_probability = float(
            torch.softmax(
                model(image.unsqueeze(0).to(device)),
                dim=1,
            )[0, target_class_id].item()
        )

    height = int(image.shape[-2])
    width = int(image.shape[-1])
    row_positions = sliding_positions(
        length=height,
        patch_size=patch_size,
        stride=stride,
    )
    column_positions = sliding_positions(
        length=width,
        patch_size=patch_size,
        stride=stride,
    )

    importance_sum = np.zeros((height, width), dtype=np.float64)
    signed_change_sum = np.zeros((height, width), dtype=np.float64)
    coverage_count = np.zeros((height, width), dtype=np.float64)
    pending_images: list[Tensor] = []
    pending_windows: list[tuple[int, int, int, int]] = []
    evaluated_windows = 0

    def evaluate_pending() -> None:
        nonlocal evaluated_windows
        if not pending_images:
            return
        batch = torch.stack(pending_images).to(
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            probabilities = torch.softmax(
                model(batch),
                dim=1,
            )[:, target_class_id].detach().cpu().numpy()

        for probability, window in zip(probabilities, pending_windows):
            top, bottom, left, right = window
            signed_change = original_probability - float(probability)
            importance_sum[top:bottom, left:right] += max(
                signed_change,
                0.0,
            )
            signed_change_sum[top:bottom, left:right] += signed_change
            coverage_count[top:bottom, left:right] += 1.0
            evaluated_windows += 1

        pending_images.clear()
        pending_windows.clear()

    for top in row_positions:
        for left in column_positions:
            bottom = min(top + patch_size, height)
            right = min(left + patch_size, width)
            pending_images.append(
                apply_good_die_occlusion(
                    image,
                    top=top,
                    left=left,
                    patch_size=patch_size,
                )
            )
            pending_windows.append((top, bottom, left, right))
            if len(pending_images) >= batch_size:
                evaluate_pending()
    evaluate_pending()

    safe_coverage = np.where(coverage_count > 0, coverage_count, 1.0)
    importance_map = importance_sum / safe_coverage
    signed_probability_change = signed_change_sum / safe_coverage
    maximum_importance = float(importance_map.max())
    if maximum_importance > 1e-12:
        importance_map = importance_map / maximum_importance

    return OcclusionResult(
        original_probability=original_probability,
        importance_map=importance_map.astype(np.float32),
        signed_probability_change=signed_probability_change.astype(
            np.float32
        ),
        evaluated_windows=evaluated_windows,
    )


def tensor_to_wafer_map(image_tensor: Tensor) -> np.ndarray:
    return np.argmax(
        image_tensor.detach().cpu().numpy(),
        axis=0,
    ).astype(np.uint8)
