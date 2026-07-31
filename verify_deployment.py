from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "artifacts/review_bundle"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manifest_path = BUNDLE / "REVIEW_BUNDLE_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

assert manifest["status"] == "FROZEN"
assert manifest["source_input_wafer_count"] == 1000
assert manifest["review_queue_records"] == 205
assert manifest["route_counts"] == {
    "ENGINEER_REVIEW": 147,
    "ABSTAIN": 58,
}
assert manifest["monitoring_status"] == "GREEN"
assert manifest["governance"]["locked_test_used"] is False

for raw_line in (BUNDLE / "SHA256SUMS.txt").read_text(
    encoding="utf-8"
).splitlines():
    if not raw_line.strip():
        continue
    expected, relative = raw_line.split("  ", 1)
    target = BUNDLE / relative
    assert target.is_file(), relative
    assert sha256_file(target) == expected, relative

model_path = ROOT / manifest["model"]["path"]
policy_path = ROOT / manifest["policy"]["path"]
assert sha256_file(model_path) == manifest["model"]["sha256"]
assert sha256_file(policy_path) == manifest["policy"]["sha256"]

input_paths = [ROOT / item["path"] for item in manifest["input_files"]]
assert len(input_paths) == 1000
assert all(path.is_file() for path in input_paths)
assert all(
    sha256_file(path) == item["sha256"]
    for path, item in zip(input_paths, manifest["input_files"])
)

step04 = load_module(
    ROOT / "scripts/04_preprocess_wafer_maps.py",
    "cloud_step04",
)
step14 = load_module(
    ROOT / "scripts/14_ablate_weighted_cross_entropy.py",
    "cloud_step14",
)
step17 = load_module(
    ROOT / "scripts/17_run_production_inference.py",
    "cloud_step17",
)

wafers = step17.load_input_files(
    [input_paths[0]],
    npz_key=None,
    maximum_wafers=1,
)
tensors, _diagnostics = step17.preprocess_wafers(
    wafers,
    step04=step04,
    maximum_side=512,
)
assert tensors.shape == (1, 3, 96, 96)

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=False,
)
class_names = [str(value) for value in manifest["model"]["class_names"]]
model = step14.BaselineCNN(
    num_classes=len(class_names),
    dropout=float(
        checkpoint.get("training_arguments", {}).get("dropout", 0.30)
    ),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.inference_mode():
    probability = torch.softmax(
        model(torch.from_numpy(tensors.astype(np.float32))),
        dim=1,
    )[0]
predicted_id = int(torch.argmax(probability).item())
predicted_name = class_names[predicted_id]
confidence = float(probability[predicted_id].item())

with (ROOT / "artifacts/source_inference_run/inference_results.csv").open(
    encoding="utf-8"
) as handle:
    rows = list(csv.DictReader(handle))
expected = next(row for row in rows if row["input_id"] == wafers[0].input_id)

assert predicted_name == expected["predicted_class_name"]
assert abs(confidence - float(expected["confidence"])) <= 1e-5

route_counts = {}
for row in rows:
    route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1
assert route_counts == {
    "AUTO_ACCEPT": 795,
    "ENGINEER_REVIEW": 147,
    "ABSTAIN": 58,
}

print("=" * 72)
print("KAVACH AI — CLOUD PACKAGE VERIFICATION PASSED")
print("=" * 72)
print("Inputs verified: 1,000")
print("Review queue: 205")
print("Routes: AUTO_ACCEPT=795, ENGINEER_REVIEW=147, ABSTAIN=58")
print(f"Prediction parity case: {wafers[0].input_id}")
print(f"Prediction: {predicted_name} ({confidence:.6f})")
print("Locked test reused: NO")
print("Model weights changed: NO")
