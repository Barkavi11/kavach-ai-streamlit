from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


VALID_DIE_VALUES = frozenset({0, 1, 2})
EXPECTED_RESOLUTION = (96, 96)
EXPECTED_CHANNELS = 3


class InferencePipelineError(RuntimeError):
    """Raised when governed serving inference cannot proceed."""


@dataclass(frozen=True)
class LoadedWafer:
    input_id: str
    source_file: str
    source_index: int
    categorical_map: np.ndarray


def normalise_array_to_maps(
    array: np.ndarray,
    *,
    source_name: str,
) -> list[np.ndarray]:
    if not isinstance(array, np.ndarray):
        raise InferencePipelineError(f"{source_name}: expected NumPy array.")
    if array.dtype == object:
        raise InferencePipelineError(
            f"{source_name}: object arrays are rejected."
        )
    if array.ndim == 2:
        return [array]
    if array.ndim == 3:
        return [array[index] for index in range(array.shape[0])]
    raise InferencePipelineError(
        f"{source_name}: expected (H,W) or (N,H,W), got {array.shape}."
    )


def load_npy(path: Path) -> list[np.ndarray]:
    return normalise_array_to_maps(
        np.load(path, allow_pickle=False),
        source_name=path.name,
    )


def load_npz(
    path: Path,
    *,
    requested_key: str | None,
) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if requested_key is not None:
            if requested_key not in keys:
                raise InferencePipelineError(
                    f"{path.name}: NPZ key not found: {requested_key}"
                )
            key = requested_key
        elif "wafer_maps" in keys:
            key = "wafer_maps"
        elif "maps" in keys:
            key = "maps"
        elif len(keys) == 1:
            key = keys[0]
        else:
            raise InferencePipelineError(
                f"{path.name}: ambiguous NPZ arrays {keys}."
            )
        array = np.asarray(archive[key])
    return normalise_array_to_maps(array, source_name=f"{path.name}:{key}")


def load_csv(path: Path) -> list[np.ndarray]:
    array = np.loadtxt(path, delimiter=",")
    return normalise_array_to_maps(array, source_name=path.name)


def load_input_files(
    files: list[Path],
    *,
    npz_key: str | None,
    maximum_wafers: int,
) -> list[LoadedWafer]:
    loaded: list[LoadedWafer] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            maps = load_npy(path)
        elif suffix == ".npz":
            maps = load_npz(path, requested_key=npz_key)
        elif suffix == ".csv":
            maps = load_csv(path)
        else:
            raise InferencePipelineError(
                f"Unsupported input extension: {path.suffix}"
            )

        for source_index, categorical_map in enumerate(maps):
            loaded.append(
                LoadedWafer(
                    input_id=f"{path.stem}:{source_index}",
                    source_file=str(path),
                    source_index=source_index,
                    categorical_map=categorical_map,
                )
            )
            if len(loaded) > maximum_wafers:
                raise InferencePipelineError("Input exceeds maximum_wafers.")

    if not loaded:
        raise InferencePipelineError("No wafer maps were loaded.")
    return loaded


def validate_categorical_map(
    categorical_map: np.ndarray,
    *,
    input_id: str,
    maximum_side: int,
) -> np.ndarray:
    if categorical_map.ndim != 2:
        raise InferencePipelineError(
            f"{input_id}: wafer map must be two-dimensional."
        )
    rows, columns = categorical_map.shape
    if rows <= 0 or columns <= 0:
        raise InferencePipelineError(f"{input_id}: wafer map is empty.")
    if rows > maximum_side or columns > maximum_side:
        raise InferencePipelineError(
            f"{input_id}: shape exceeds maximum side {maximum_side}."
        )
    numeric = np.asarray(categorical_map, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise InferencePipelineError(
            f"{input_id}: NaN or infinite values are not allowed."
        )
    if not np.all(numeric == np.round(numeric)):
        raise InferencePipelineError(
            f"{input_id}: values must be integers 0, 1, or 2."
        )
    categorical = numeric.astype(np.uint8)
    unique_values = {int(value) for value in np.unique(categorical)}
    if not unique_values.issubset(VALID_DIE_VALUES):
        raise InferencePipelineError(
            f"{input_id}: invalid values {sorted(unique_values)}."
        )
    if int(np.count_nonzero(categorical > 0)) == 0:
        raise InferencePipelineError(f"{input_id}: no active dies.")
    return categorical


def preprocess_wafers(
    wafers: list[LoadedWafer],
    *,
    step04,
    maximum_side: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tensors = np.empty(
        (
            len(wafers),
            EXPECTED_CHANNELS,
            EXPECTED_RESOLUTION[0],
            EXPECTED_RESOLUTION[1],
        ),
        dtype=np.uint8,
    )
    diagnostics: list[dict[str, Any]] = []

    for index, wafer in enumerate(wafers):
        categorical = validate_categorical_map(
            wafer.categorical_map,
            input_id=wafer.input_id,
            maximum_side=maximum_side,
        )
        processed_map, transform = step04.resize_and_pad_categorical(
            categorical,
            target_height=EXPECTED_RESOLUTION[0],
            target_width=EXPECTED_RESOLUTION[1],
        )
        tensor = step04.one_hot_encode_wafer(processed_map)
        if tensor.shape != (
            EXPECTED_CHANNELS,
            *EXPECTED_RESOLUTION,
        ):
            raise InferencePipelineError(
                f"{wafer.input_id}: unexpected tensor shape {tensor.shape}."
            )
        tensors[index] = tensor
        source_active = int(np.count_nonzero(categorical > 0))
        source_failed = int(np.count_nonzero(categorical == 2))
        diagnostics.append(
            {
                "input_id": wafer.input_id,
                "source_height": int(categorical.shape[0]),
                "source_width": int(categorical.shape[1]),
                "source_active_dies": source_active,
                "source_failed_dies": source_failed,
                "source_failed_ratio": source_failed / source_active,
                "preprocessing_transform": transform,
            }
        )
    return tensors, diagnostics
