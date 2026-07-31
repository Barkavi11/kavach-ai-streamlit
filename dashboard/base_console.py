from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st
import torch


ROOT = Path(__file__).resolve().parents[1]
LOSS_NAME = "sqrt_inverse_weighted_cross_entropy"

STEP04 = ROOT / "scripts/04_preprocess_wafer_maps.py"
STEP11 = ROOT / "scripts/11_validate_explanations_with_occlusion.py"
STEP14 = ROOT / "scripts/14_ablate_weighted_cross_entropy.py"
STEP17 = ROOT / "scripts/17_run_production_inference.py"

BUNDLE_ROOT = ROOT / "artifacts" / "review_bundle"
DEFAULT_DATABASE = ROOT / ".runtime" / "kavach_engineer_review.sqlite3"

ACTIONS = [
    "CONFIRM_MODEL_LABEL",
    "OVERRIDE_LABEL",
    "ESCALATE_FOR_PROCESS_REVIEW",
    "MARK_DATA_QUALITY_ISSUE",
]


class ConsoleError(RuntimeError):
    """Raised when the governed review console cannot operate safely."""


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    arguments, _unknown = parser.parse_known_args(sys.argv[1:])
    return arguments


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConsoleError(f"Required JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConsoleError(f"{path} must contain a JSON object.")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_bundle_checksums(bundle: Path) -> None:
    checksum_path = bundle / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise ConsoleError(
            f"Bundle checksum manifest not found: {checksum_path}"
        )

    required_entries = {
        "REVIEW_BUNDLE_MANIFEST.json",
        "review_queue.csv",
        "evidence/inference_manifest.json",
        "evidence/MONITORING_RUN_MANIFEST.json",
        "evidence/MONITORING_DECISION.json",
        "evidence/FAITHFULNESS_DECISION.json",
        "evidence/STEP19B_MANIFEST.json",
    }

    seen: set[str] = set()
    bundle_root = bundle.resolve()

    for line_number, raw_line in enumerate(
        checksum_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            expected_hash, relative_name = raw_line.split(
                "  ",
                1,
            )
        except ValueError as exc:
            raise ConsoleError(
                f"Invalid SHA256SUMS entry at line {line_number}."
            ) from exc

        relative_path = Path(relative_name)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ConsoleError(
                f"Unsafe checksum path: {relative_name}"
            )

        target = (bundle_root / relative_path).resolve()

        if target != bundle_root and bundle_root not in target.parents:
            raise ConsoleError(
                f"Checksum path escapes bundle: {relative_name}"
            )

        if not target.is_file():
            raise ConsoleError(
                f"Checksummed file is missing: {relative_name}"
            )

        if sha256_file(target) != expected_hash:
            raise ConsoleError(
                f"Bundle checksum mismatch: {relative_name}"
            )

        seen.add(relative_path.as_posix())

    missing = required_entries.difference(seen)
    if missing:
        raise ConsoleError(
            "SHA256SUMS is missing governed files: "
            f"{sorted(missing)}"
        )


def load_module(path: Path, name: str):
    if not path.is_file():
        raise ConsoleError(f"Project script not found: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ConsoleError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def resolve_bundle(explicit: Path | None) -> Path:
    if explicit is not None:
        bundle = explicit.expanduser().resolve()
    else:
        bundle = BUNDLE_ROOT.expanduser().resolve()

    if not bundle.is_dir():
        raise ConsoleError(f"Review bundle not found: {bundle}")
    return bundle


def validate_bundle(
    bundle: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = bundle / "REVIEW_BUNDLE_MANIFEST.json"
    queue_path = bundle / "review_queue.csv"

    if not manifest_path.is_file() or not queue_path.is_file():
        raise ConsoleError("Review bundle is incomplete.")

    verify_bundle_checksums(bundle)
    manifest = load_json(manifest_path)
    if manifest.get("step") != "20A":
        raise ConsoleError("Bundle is not a Step 20A output.")
    if manifest.get("status") != "FROZEN":
        raise ConsoleError("Review bundle status must be FROZEN.")
    if manifest.get("governance", {}).get("locked_test_used") is not False:
        raise ConsoleError("Review-bundle locked-test marker is invalid.")

    explanation = manifest.get("explanation_policy", {})
    if explanation.get("approved_method") != "good_die_occlusion":
        raise ConsoleError("Approved explanation method is invalid.")
    if explanation.get("gradcam") != "hidden":
        raise ConsoleError("Grad-CAM must remain hidden.")
    if explanation.get("consensus") != "hidden":
        raise ConsoleError("Consensus maps must remain hidden.")
    if explanation.get("physical_root_cause_claimed") is not False:
        raise ConsoleError("Physical root-cause claims are prohibited.")

    checkpoint = Path(str(manifest["model"]["path"])).expanduser().resolve()
    policy = Path(str(manifest["policy"]["path"])).expanduser().resolve()

    if not checkpoint.is_file() or not policy.is_file():
        raise ConsoleError("Frozen model or policy is missing.")
    if sha256_file(checkpoint) != str(manifest["model"]["sha256"]):
        raise ConsoleError("Frozen checkpoint checksum mismatch.")
    if sha256_file(policy) != str(manifest["policy"]["sha256"]):
        raise ConsoleError("Frozen policy checksum mismatch.")

    for entry in manifest["input_files"]:
        source = Path(str(entry["path"])).expanduser().resolve()
        if not source.is_file():
            raise ConsoleError(f"Input file is missing: {source}")
        if sha256_file(source) != str(entry["sha256"]):
            raise ConsoleError(f"Input checksum mismatch: {source}")

    queue = pd.read_csv(queue_path)
    if len(queue) != int(manifest["review_queue_records"]):
        raise ConsoleError("Queue count differs from bundle manifest.")
    if queue["input_id"].duplicated().any():
        raise ConsoleError("Review queue input IDs are not unique.")
    if not queue["effective_route"].astype(str).isin(
        ["ENGINEER_REVIEW", "ABSTAIN"]
    ).all():
        raise ConsoleError("Queue contains a non-review route.")

    return manifest, queue


def initialise_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS review_events (
                sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                created_at_utc TEXT NOT NULL,
                bundle_id TEXT NOT NULL,
                inference_run_id TEXT NOT NULL,
                input_id TEXT NOT NULL,
                engineer_id TEXT NOT NULL,
                action TEXT NOT NULL,
                model_label TEXT NOT NULL,
                selected_label TEXT,
                notes TEXT NOT NULL,
                monitoring_status TEXT NOT NULL,
                effective_route TEXT NOT NULL,
                model_sha256 TEXT NOT NULL,
                policy_sha256 TEXT NOT NULL,
                bundle_manifest_sha256 TEXT NOT NULL,
                case_row_sha256 TEXT NOT NULL,
                supersedes_event_id TEXT,
                previous_chain_hash TEXT,
                event_hash TEXT NOT NULL UNIQUE
            );

            CREATE INDEX IF NOT EXISTS idx_review_events_case
            ON review_events(bundle_id, input_id, sequence_id);

            CREATE TRIGGER IF NOT EXISTS prevent_review_event_update
            BEFORE UPDATE ON review_events
            BEGIN
                SELECT RAISE(ABORT, 'review_events is append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_review_event_delete
            BEFORE DELETE ON review_events
            BEGIN
                SELECT RAISE(ABORT, 'review_events is append-only');
            END;
            """
        )
        connection.commit()


def verify_chain(path: Path) -> tuple[bool, str]:
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT
                event_id,
                created_at_utc,
                bundle_id,
                inference_run_id,
                input_id,
                engineer_id,
                action,
                model_label,
                selected_label,
                notes,
                monitoring_status,
                effective_route,
                model_sha256,
                policy_sha256,
                bundle_manifest_sha256,
                case_row_sha256,
                supersedes_event_id,
                previous_chain_hash,
                event_hash
            FROM review_events
            ORDER BY sequence_id
            """
        ).fetchall()

    field_names = [
        "event_id",
        "created_at_utc",
        "bundle_id",
        "inference_run_id",
        "input_id",
        "engineer_id",
        "action",
        "model_label",
        "selected_label",
        "notes",
        "monitoring_status",
        "effective_route",
        "model_sha256",
        "policy_sha256",
        "bundle_manifest_sha256",
        "case_row_sha256",
        "supersedes_event_id",
        "previous_chain_hash",
    ]

    previous_hash: str | None = None
    for row in rows:
        event = dict(zip(field_names, row[:-1]))
        stored_hash = str(row[-1])

        if event["previous_chain_hash"] != previous_hash:
            return False, f"Chain-link mismatch at {event['event_id']}."
        if canonical_hash(event) != stored_hash:
            return False, f"Event-hash mismatch at {event['event_id']}."

        previous_hash = stored_hash

    return True, f"{len(rows)} event(s) verified."


def latest_events(path: Path, bundle_id: str) -> pd.DataFrame:
    with closing(sqlite3.connect(path)) as connection:
        return pd.read_sql_query(
            """
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY bundle_id, input_id
                        ORDER BY sequence_id DESC
                    ) AS rank_value
                FROM review_events
                WHERE bundle_id = ?
            )
            SELECT *
            FROM ranked
            WHERE rank_value = 1
            """,
            connection,
            params=(bundle_id,),
        )


def append_event(
    path: Path,
    *,
    manifest: dict[str, Any],
    manifest_hash: str,
    case: pd.Series,
    engineer_id: str,
    action: str,
    selected_label: str | None,
    notes: str,
) -> str:
    case_payload: dict[str, Any] = {}
    for key, value in case.to_dict().items():
        if pd.isna(value):
            case_payload[key] = None
        elif isinstance(value, np.generic):
            case_payload[key] = value.item()
        else:
            case_payload[key] = value

    case_hash = canonical_hash(case_payload)

    with closing(sqlite3.connect(path, timeout=10.0)) as connection:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN IMMEDIATE")

        prior_chain = connection.execute(
            """
            SELECT event_hash
            FROM review_events
            ORDER BY sequence_id DESC
            LIMIT 1
            """
        ).fetchone()
        previous_chain_hash = (
            None if prior_chain is None else str(prior_chain[0])
        )

        prior_case = connection.execute(
            """
            SELECT event_id
            FROM review_events
            WHERE bundle_id = ? AND input_id = ?
            ORDER BY sequence_id DESC
            LIMIT 1
            """,
            (str(manifest["bundle_id"]), str(case["input_id"])),
        ).fetchone()
        supersedes_event_id = (
            None if prior_case is None else str(prior_case[0])
        )

        event = {
            "event_id": str(uuid.uuid4()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_id": str(manifest["bundle_id"]),
            "inference_run_id": str(
                manifest["source_inference_run_id"]
            ),
            "input_id": str(case["input_id"]),
            "engineer_id": engineer_id.strip(),
            "action": action,
            "model_label": str(case["predicted_class_name"]),
            "selected_label": selected_label,
            "notes": notes.strip(),
            "monitoring_status": str(case["monitoring_status"]),
            "effective_route": str(case["effective_route"]),
            "model_sha256": str(manifest["model"]["sha256"]),
            "policy_sha256": str(manifest["policy"]["sha256"]),
            "bundle_manifest_sha256": manifest_hash,
            "case_row_sha256": case_hash,
            "supersedes_event_id": supersedes_event_id,
            "previous_chain_hash": previous_chain_hash,
        }
        event_hash = canonical_hash(event)

        connection.execute(
            """
            INSERT INTO review_events (
                event_id,
                created_at_utc,
                bundle_id,
                inference_run_id,
                input_id,
                engineer_id,
                action,
                model_label,
                selected_label,
                notes,
                monitoring_status,
                effective_route,
                model_sha256,
                policy_sha256,
                bundle_manifest_sha256,
                case_row_sha256,
                supersedes_event_id,
                previous_chain_hash,
                event_hash
            ) VALUES (
                :event_id,
                :created_at_utc,
                :bundle_id,
                :inference_run_id,
                :input_id,
                :engineer_id,
                :action,
                :model_label,
                :selected_label,
                :notes,
                :monitoring_status,
                :effective_route,
                :model_sha256,
                :policy_sha256,
                :bundle_manifest_sha256,
                :case_row_sha256,
                :supersedes_event_id,
                :previous_chain_hash,
                :event_hash
            )
            """,
            {**event, "event_hash": event_hash},
        )
        connection.commit()

    return str(event["event_id"])


@st.cache_resource(show_spinner=False)
def load_runtime(bundle_path: str, manifest_hash: str):
    bundle = Path(bundle_path)
    manifest, queue = validate_bundle(bundle)

    manifest_path = bundle / "REVIEW_BUNDLE_MANIFEST.json"
    if sha256_file(manifest_path) != manifest_hash:
        raise ConsoleError("Bundle manifest changed during load.")

    step04 = load_module(STEP04, "kavach_step04_console")
    step11 = load_module(STEP11, "kavach_step11_console")
    step14 = load_module(STEP14, "kavach_step14_console")
    step17 = load_module(STEP17, "kavach_step17_console")

    input_files = [
        Path(str(entry["path"])).expanduser().resolve()
        for entry in manifest["input_files"]
    ]
    expected_wafer_count = int(
        manifest["source_input_wafer_count"]
    )

    wafers = step17.load_input_files(
        input_files,
        npz_key=None,
        maximum_wafers=expected_wafer_count,
    )

    if len(wafers) != expected_wafer_count:
        raise ConsoleError(
            "Reloaded wafer count differs from the frozen bundle manifest."
        )

    tensors, _diagnostics = step17.preprocess_wafers(
        wafers,
        step04=step04,
        maximum_side=512,
    )
    lookup = {
        str(wafer.input_id): index
        for index, wafer in enumerate(wafers)
    }
    if len(lookup) != len(wafers):
        raise ConsoleError("Runtime input IDs are not unique.")

    run_device = (
        torch.device("mps")
        if hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    checkpoint_path = Path(
        str(manifest["model"]["path"])
    ).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=run_device,
        weights_only=False,
    )
    class_names = [str(value) for value in manifest["model"]["class_names"]]
    if [str(value) for value in checkpoint["class_names"]] != class_names:
        raise ConsoleError(
            "Checkpoint and review-bundle class orders differ."
        )

    model = step14.BaselineCNN(
        num_classes=len(class_names),
        dropout=float(
            checkpoint.get("training_arguments", {}).get(
                "dropout",
                0.30,
            )
        ),
    ).to(run_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return manifest, queue, tensors, lookup, model, run_device, step11


def all_events(path: Path, bundle_id: str) -> pd.DataFrame:
    """Read the complete immutable event history for one review bundle."""
    with closing(sqlite3.connect(path)) as connection:
        return pd.read_sql_query(
            """
            SELECT
                sequence_id,
                event_id,
                created_at_utc,
                bundle_id,
                inference_run_id,
                input_id,
                engineer_id,
                action,
                model_label,
                selected_label,
                notes,
                monitoring_status,
                effective_route,
                supersedes_event_id,
                previous_chain_hash,
                event_hash
            FROM review_events
            WHERE bundle_id = ?
            ORDER BY sequence_id DESC
            """,
            connection,
            params=(bundle_id,),
        )


def inject_styles() -> None:
    """Apply an Infineon-inspired light semiconductor operations theme."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --ifx-teal: #008578;
            --ifx-teal-dark: #00675f;
            --ifx-teal-soft: #e6f4f2;
            --ifx-blue: #005b7f;
            --ifx-magenta: #d5006d;
            --ifx-magenta-soft: #fbe7f1;
            --ifx-amber: #e28a00;
            --ifx-red: #c62828;
            --ifx-green: #248a5a;
            --ifx-ink: #18191b;
            --ifx-muted: #5d6367;
            --ifx-line: #d9dddc;
            --ifx-line-strong: #bfc6c4;
            --ifx-page: #f5f6f4;
            --ifx-card: #ffffff;
            --ifx-card-alt: #f9faf8;
        }

        html,
        body,
        [data-testid="stAppViewContainer"],
        .stApp {
            color-scheme: light !important;
            background: var(--ifx-page) !important;
            color: var(--ifx-ink) !important;
        }

        /*
         * Keep Streamlit's native sidebar controls available.
         * When the panel is open, Streamlit shows <<.
         * When it is collapsed, Streamlit shows >>.
         * Only the unrelated toolbar/menu elements are hidden.
         */
        header[data-testid="stHeader"] {
            display: flex !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
        }

        .block-container {
            max-width: 1560px;
            padding-top: 1.15rem;
            padding-bottom: 3rem;
        }

        p,
        label,
        span,
        div {
            font-family:
                Inter,
                "Segoe UI",
                Arial,
                sans-serif;
        }

        h1,
        h2,
        h3,
        h4 {
            color: var(--ifx-ink) !important;
            letter-spacing: -0.018em;
        }

        [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid var(--ifx-line);
            box-shadow: 5px 0 18px rgba(26, 44, 42, 0.04);
        }

        [data-testid="stSidebar"] > div:first-child {
            background: #ffffff !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.1rem;
            padding-left: 1.15rem;
            padding-right: 1.15rem;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--ifx-line) !important;
        }

        [data-testid="stMetric"] {
            min-height: 112px;
            background: var(--ifx-card) !important;
            border: 1px solid var(--ifx-line);
            border-top: 3px solid var(--ifx-teal);
            border-radius: 2px;
            padding: 0.95rem 1rem;
            box-shadow: 0 7px 20px rgba(31, 50, 47, 0.06);
        }

        [data-testid="stMetricLabel"] {
            color: var(--ifx-muted) !important;
            font-size: 0.79rem !important;
            font-weight: 650 !important;
            letter-spacing: 0.01em;
        }

        [data-testid="stMetricValue"] {
            color: var(--ifx-ink) !important;
            font-weight: 650 !important;
            line-height: 1.05 !important;
        }

        [data-testid="stMetricDelta"] {
            color: var(--ifx-teal-dark) !important;
        }

        [data-testid="stForm"] {
            background: var(--ifx-card) !important;
            border: 1px solid var(--ifx-line);
            border-radius: 2px;
            padding: 1.15rem 1.1rem 0.35rem 1.1rem;
            box-shadow: 0 7px 20px rgba(31, 50, 47, 0.05);
        }

        [data-testid="stDataFrame"] {
            background: var(--ifx-card) !important;
            border: 1px solid var(--ifx-line);
            border-radius: 2px;
            overflow: hidden;
            box-shadow: 0 7px 20px rgba(31, 50, 47, 0.04);
        }

        [data-testid="stExpander"] {
            background: var(--ifx-card) !important;
            border: 1px solid var(--ifx-line) !important;
            border-radius: 2px !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="base-input"] {
            background: #ffffff !important;
            border-color: var(--ifx-line-strong) !important;
            border-radius: 2px !important;
            color: var(--ifx-ink) !important;
            box-shadow: none !important;
        }

        input,
        textarea,
        [role="combobox"] {
            color: var(--ifx-ink) !important;
            caret-color: var(--ifx-teal) !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #8a9190 !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {
            background: #ffffff !important;
            color: var(--ifx-ink) !important;
        }

        li[role="option"] {
            color: var(--ifx-ink) !important;
        }

        li[role="option"]:hover {
            background: var(--ifx-teal-soft) !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            min-height: 2.75rem;
            border-radius: 2px !important;
            border: 1px solid var(--ifx-teal) !important;
            background: #ffffff !important;
            color: var(--ifx-teal-dark) !important;
            font-weight: 700 !important;
            box-shadow: none !important;
            transition:
                background 120ms ease,
                color 120ms ease,
                border-color 120ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--ifx-teal-soft) !important;
            border-color: var(--ifx-teal-dark) !important;
            color: var(--ifx-teal-dark) !important;
        }

        button[kind="primary"],
        [data-testid="stFormSubmitButton"] button {
            background: var(--ifx-teal) !important;
            color: #ffffff !important;
            border-color: var(--ifx-teal) !important;
        }

        button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: var(--ifx-teal-dark) !important;
            border-color: var(--ifx-teal-dark) !important;
        }

        button:disabled {
            opacity: 0.42 !important;
        }

        [data-testid="stCheckbox"] label span {
            color: var(--ifx-ink) !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            background: var(--ifx-teal) !important;
            border-color: var(--ifx-teal) !important;
        }

        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background: var(--ifx-teal) !important;
        }

        div[data-testid="stProgress"] > div {
            background: #e8eceb !important;
            border-radius: 0 !important;
        }

        div[data-testid="stProgress"] > div > div > div {
            background: linear-gradient(
                90deg,
                var(--ifx-teal),
                #40a99d
            ) !important;
            border-radius: 0 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1.6rem;
            border-bottom: 1px solid var(--ifx-line);
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--ifx-muted) !important;
            border-radius: 0 !important;
            padding-left: 0;
            padding-right: 0;
            font-weight: 650;
        }

        .stTabs [aria-selected="true"] {
            color: var(--ifx-teal-dark) !important;
            border-bottom-color: var(--ifx-teal) !important;
        }

        .ifx-sidebar-brand {
            position: relative;
            margin: -0.2rem -0.15rem 0.85rem -0.15rem;
            padding: 1rem 0.9rem 1rem 1rem;
            background: #ffffff;
            border: 1px solid var(--ifx-line);
            border-left: 5px solid var(--ifx-teal);
        }

        .ifx-sidebar-brand::after {
            content: "";
            position: absolute;
            right: 0;
            top: 0;
            width: 7px;
            height: 100%;
            background: var(--ifx-magenta);
        }

        .ifx-brand-name {
            color: var(--ifx-ink);
            font-size: 1.28rem;
            line-height: 1;
            font-weight: 760;
            letter-spacing: -0.02em;
        }

        .ifx-brand-name span {
            color: var(--ifx-teal);
        }

        .ifx-brand-subtitle {
            margin-top: 0.35rem;
            color: var(--ifx-muted);
            font-size: 0.72rem;
            font-weight: 650;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .ifx-hero {
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(300px, 0.92fr);
            min-height: 250px;
            margin-bottom: 1.15rem;
            overflow: hidden;
            border: 1px solid var(--ifx-line);
            background: #ffffff;
            box-shadow: 0 9px 28px rgba(31, 50, 47, 0.07);
        }

        .ifx-hero-copy {
            position: relative;
            z-index: 2;
            padding: 2.25rem 2.4rem 2rem 2.4rem;
        }

        .ifx-eyebrow {
            color: var(--ifx-teal-dark);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .ifx-title {
            max-width: 760px;
            margin: 0.55rem 0 0;
            color: var(--ifx-ink);
            font-size: clamp(2rem, 3vw, 3.2rem);
            line-height: 1.03;
            font-weight: 680;
            letter-spacing: -0.045em;
        }

        .ifx-subtitle {
            max-width: 690px;
            margin-top: 1rem;
            color: var(--ifx-muted);
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .ifx-hero-visual {
            position: relative;
            min-height: 250px;
            background:
                radial-gradient(
                    circle at 69% 36%,
                    rgba(255, 255, 255, 0.23) 0 3px,
                    transparent 4px
                ) 0 0 / 24px 24px,
                radial-gradient(
                    circle at 35% 62%,
                    rgba(255, 255, 255, 0.13) 0 2px,
                    transparent 3px
                ) 0 0 / 17px 17px,
                linear-gradient(
                    135deg,
                    #00a294 0%,
                    #008578 48%,
                    #00637a 100%
                );
            clip-path: polygon(18% 0, 100% 0, 100% 100%, 0 100%);
        }

        .ifx-hero-visual::before {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            right: 9%;
            top: 18px;
            border: 2px solid rgba(255, 255, 255, 0.50);
            border-radius: 50%;
            background:
                repeating-radial-gradient(
                    circle,
                    transparent 0 14px,
                    rgba(255, 255, 255, 0.18) 15px 16px
                ),
                repeating-linear-gradient(
                    90deg,
                    rgba(255, 255, 255, 0.15) 0 1px,
                    transparent 1px 18px
                ),
                repeating-linear-gradient(
                    0deg,
                    rgba(255, 255, 255, 0.15) 0 1px,
                    transparent 1px 18px
                );
            box-shadow:
                0 0 0 18px rgba(255, 255, 255, 0.06),
                0 0 50px rgba(0, 0, 0, 0.12);
        }

        .ifx-hero-visual::after {
            content: "WAFER INTELLIGENCE";
            position: absolute;
            right: 2rem;
            bottom: 1.35rem;
            color: rgba(255, 255, 255, 0.90);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.17em;
        }

        .ifx-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1.35rem;
        }

        .kavach-badge {
            display: inline-flex;
            align-items: center;
            min-height: 1.75rem;
            padding: 0.32rem 0.62rem;
            border: 1px solid var(--ifx-line-strong);
            border-radius: 999px;
            background: #ffffff;
            color: var(--ifx-muted);
            font-size: 0.68rem;
            font-weight: 760;
            letter-spacing: 0.045em;
            white-space: nowrap;
        }

        .kavach-badge.green {
            color: var(--ifx-green);
            border-color: #94cfb1;
            background: #f0faf5;
        }

        .kavach-badge.amber {
            color: #9b6200;
            border-color: #e7c47e;
            background: #fff8e9;
        }

        .kavach-badge.red {
            color: var(--ifx-red);
            border-color: #e6a3a3;
            background: #fff2f2;
        }

        .kavach-panel {
            border: 1px solid var(--ifx-line);
            border-radius: 2px;
            background: var(--ifx-card);
            padding: 1rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 6px 18px rgba(31, 50, 47, 0.04);
        }

        .kavach-panel-title {
            color: var(--ifx-ink);
            font-size: 0.88rem;
            font-weight: 750;
            margin-bottom: 0.65rem;
        }

        .kavach-kv {
            display: grid;
            grid-template-columns: minmax(7.5rem, 0.8fr) 1.4fr;
            gap: 0.48rem 0.75rem;
            font-size: 0.80rem;
        }

        .kavach-kv .label {
            color: var(--ifx-muted);
        }

        .kavach-kv .value {
            color: var(--ifx-ink);
            font-weight: 560;
            overflow-wrap: anywhere;
        }

        .kavach-callout {
            border-left: 4px solid var(--ifx-teal);
            border-radius: 0;
            background: var(--ifx-teal-soft);
            padding: 0.78rem 0.9rem;
            color: #124d48;
            font-size: 0.82rem;
            line-height: 1.5;
            margin-top: 0.7rem;
        }

        .kavach-warning {
            border-left-color: var(--ifx-amber);
            background: #fff8e8;
            color: #6e4a08;
        }

        .kavach-small {
            color: var(--ifx-muted);
            font-size: 0.75rem;
            line-height: 1.4;
        }

        .kavach-probability-row {
            margin-bottom: 0.5rem;
        }

        .kavach-probability-label {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: var(--ifx-ink);
            font-size: 0.79rem;
            margin-bottom: 0.18rem;
        }

        .ifx-section-label {
            margin: 0.2rem 0 0.7rem;
            color: var(--ifx-teal-dark);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }

        .ifx-quick-reference {
            margin-top: 0.22rem;
            padding: 0.82rem 0.78rem 0.76rem;
            border: 1px solid var(--ifx-line);
            border-radius: 2px;
            background: #ffffff;
            box-shadow: 0 5px 16px rgba(31, 50, 47, 0.035);
        }

        .ifx-quick-reference-title {
            margin-bottom: 0.70rem;
            color: var(--ifx-teal-dark);
            font-size: 0.67rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        .ifx-quick-reference-subtitle {
            margin: 0.60rem 0 0.40rem;
            color: var(--ifx-ink);
            font-size: 0.72rem;
            font-weight: 720;
        }

        .ifx-legend-row,
        .ifx-route-row {
            display: grid;
            align-items: center;
            gap: 0.46rem;
            margin-bottom: 0.34rem;
            color: #44494b;
            font-size: 0.68rem;
            line-height: 1.28;
        }

        .ifx-legend-row {
            grid-template-columns: 0.72rem 1fr;
        }

        .ifx-route-row {
            grid-template-columns: 7.05rem minmax(0, 1fr);
        }

        .ifx-legend-dot {
            width: 0.66rem;
            height: 0.66rem;
            border-radius: 50%;
            display: inline-block;
        }

        .ifx-legend-dot.pass {
            background: var(--ifx-teal);
        }

        .ifx-legend-dot.fail {
            background: var(--ifx-magenta);
        }

        .ifx-legend-dot.inactive {
            background: #c8cdcb;
        }

        .ifx-route-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            box-sizing: border-box;
            min-height: 1.30rem;
            padding: 0.10rem 0.38rem;
            border: 1px solid var(--ifx-line-strong);
            border-radius: 999px;
            background: #ffffff;
            color: #59605f;
            font-size: 0.56rem;
            font-weight: 800;
            letter-spacing: 0.015em;
            white-space: nowrap;
        }

        .ifx-route-chip.auto {
            border-color: #8fd4bf;
            background: #eef9f5;
            color: #167650;
        }

        .ifx-route-chip.review {
            border-color: #eca5c8;
            background: #fff1f7;
            color: #b31463;
        }

        .ifx-quick-reference-note {
            margin-top: 0.70rem;
            padding-top: 0.58rem;
            border-top: 1px solid var(--ifx-line);
            color: var(--ifx-muted);
            font-size: 0.64rem;
            line-height: 1.42;
        }

        .ifx-quick-reference-note strong {
            color: var(--ifx-teal-dark);
        }

        @media (max-width: 980px) {
            .ifx-hero {
                grid-template-columns: 1fr;
            }

            .ifx-hero-visual {
                min-height: 190px;
                clip-path: polygon(0 13%, 100% 0, 100% 100%, 0 100%);
            }
        }
        
        /* Optimized workbench layout */
        .ifx-case-summary {
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(145px, 1fr));
            gap: 0.75rem;
            margin: 0.85rem 0 0.95rem;
        }

        .ifx-summary-card,
        .ifx-evidence-card {
            background: #ffffff;
            border: 1px solid var(--ifx-line);
            border-top: 3px solid var(--ifx-teal);
            padding: 0.82rem 0.9rem;
            min-width: 0;
            box-shadow: 0 5px 16px rgba(31, 50, 47, 0.045);
        }

        .ifx-summary-label,
        .ifx-evidence-label {
            color: var(--ifx-muted);
            font-size: 0.70rem;
            font-weight: 720;
            letter-spacing: 0.035em;
            text-transform: uppercase;
        }

        .ifx-summary-value {
            margin-top: 0.32rem;
            color: var(--ifx-ink);
            font-size: 1.34rem;
            line-height: 1.1;
            font-weight: 680;
            overflow-wrap: anywhere;
        }

        .ifx-evidence-grid {
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.7rem;
            margin: 0.72rem 0 0.75rem;
        }

        .ifx-evidence-value {
            margin-top: 0.30rem;
            color: var(--ifx-ink);
            font-size: 1.05rem;
            line-height: 1.25;
            font-weight: 680;
            overflow-wrap: anywhere;
        }

        .ifx-compact-status {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.45rem;
            margin: 0.6rem 0 0.2rem;
        }

        .ifx-status-line {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            padding: 0.55rem 0.65rem;
            border-left: 3px solid var(--ifx-teal);
            background: var(--ifx-teal-soft);
            color: #174d48;
            font-size: 0.76rem;
        }

        .ifx-status-line strong {
            color: var(--ifx-teal-dark);
        }

        .ifx-evidence-section {
            margin-top: 1.15rem;
            padding-top: 0.2rem;
            border-top: 1px solid var(--ifx-line);
        }

        .ifx-layout-note {
            margin: 0.25rem 0 0.7rem;
            color: var(--ifx-muted);
            font-size: 0.78rem;
            line-height: 1.45;
        }

        /* Force form controls to remain light */
        [data-testid="stForm"] input,
        [data-testid="stForm"] textarea,
        [data-testid="stForm"] [role="combobox"],
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [role="combobox"] {
            background: #ffffff !important;
            color: var(--ifx-ink) !important;
            -webkit-text-fill-color: var(--ifx-ink) !important;
        }

        [data-testid="stForm"] input:disabled {
            background: #f0f2f1 !important;
            color: #4e5554 !important;
            -webkit-text-fill-color: #4e5554 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            box-shadow: none !important;
        }

        @media (max-width: 1120px) {
            .ifx-case-summary {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
        }

        @media (max-width: 760px) {
            .ifx-case-summary,
            .ifx-evidence-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        /* V3: balanced, full-width evidence architecture */
        .ifx-hero {
            min-height: 215px;
            grid-template-columns:
                minmax(0, 1.15fr) minmax(280px, 0.85fr);
        }

        .ifx-hero-copy {
            padding: 1.65rem 2rem 1.55rem 2rem;
        }

        .ifx-title {
            font-size: clamp(2rem, 2.7vw, 2.75rem);
        }

        .ifx-hero-visual {
            min-height: 215px;
        }

        .ifx-hero-visual::before {
            width: 170px;
            height: 170px;
            top: 18px;
        }

        .ifx-primary-metrics {
            display: grid;
            grid-template-columns:
                repeat(4, minmax(0, 1fr));
            gap: 0.72rem;
            margin: 0.85rem 0 0.65rem;
        }

        .ifx-primary-card {
            min-width: 0;
            background: #ffffff;
            border: 1px solid var(--ifx-line);
            border-top: 3px solid var(--ifx-teal);
            padding: 0.82rem 0.88rem;
            box-shadow: 0 5px 16px rgba(31, 50, 47, 0.045);
        }

        .ifx-primary-label {
            color: var(--ifx-muted);
            font-size: 0.68rem;
            font-weight: 760;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .ifx-primary-value {
            margin-top: 0.28rem;
            color: var(--ifx-ink);
            font-size: 1.30rem;
            line-height: 1.12;
            font-weight: 690;
            overflow-wrap: anywhere;
        }

        .ifx-case-state-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
            margin: 0.2rem 0 0.9rem;
        }

        .ifx-state-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.30rem 0.62rem;
            border: 1px solid var(--ifx-line-strong);
            background: #ffffff;
            color: var(--ifx-muted);
            border-radius: 999px;
            font-size: 0.69rem;
            font-weight: 760;
            letter-spacing: 0.035em;
        }

        .ifx-state-chip.route {
            border-color: #e7c47e;
            background: #fff8e9;
            color: #8a5900;
        }

        .ifx-state-chip.reviewed {
            border-color: #94cfb1;
            background: #f0faf5;
            color: var(--ifx-green);
        }

        .ifx-state-chip.pending {
            border-color: #bfc6c4;
            background: #f7f8f7;
            color: #515958;
        }

        .ifx-wafer-metrics {
            display: grid;
            grid-template-columns:
                repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.72rem 0 0.72rem;
        }

        .ifx-wafer-metric {
            min-width: 0;
            background: #ffffff;
            border: 1px solid var(--ifx-line);
            border-left: 4px solid var(--ifx-teal);
            padding: 0.75rem 0.82rem;
        }

        .ifx-wafer-metric-label {
            color: var(--ifx-muted);
            font-size: 0.67rem;
            font-weight: 750;
            letter-spacing: 0.035em;
            text-transform: uppercase;
        }

        .ifx-wafer-metric-value {
            margin-top: 0.26rem;
            color: var(--ifx-ink);
            font-size: 1.03rem;
            line-height: 1.25;
            font-weight: 680;
            overflow-wrap: anywhere;
        }

        .ifx-decision-evidence {
            margin-top: 1.35rem;
            padding-top: 1.15rem;
            border-top: 1px solid var(--ifx-line);
        }

        .ifx-alt-card {
            height: 100%;
            background: #ffffff;
            border: 1px solid var(--ifx-line);
            border-top: 3px solid var(--ifx-teal);
            padding: 0.92rem 1rem;
            box-shadow: 0 5px 16px rgba(31, 50, 47, 0.04);
        }

        .ifx-alt-rank {
            color: var(--ifx-teal-dark);
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .ifx-alt-header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.8rem;
            margin: 0.35rem 0 0.55rem;
        }

        .ifx-alt-class {
            min-width: 0;
            color: var(--ifx-ink);
            font-size: 1.04rem;
            line-height: 1.2;
            font-weight: 700;
            overflow-wrap: anywhere;
        }

        .ifx-alt-probability {
            flex: 0 0 auto;
            color: var(--ifx-teal-dark);
            font-size: 1rem;
            font-weight: 760;
        }

        .ifx-native-progress {
            height: 0.48rem;
            overflow: hidden;
            background: #e9edec;
        }

        .ifx-native-progress > div {
            height: 100%;
            background: linear-gradient(
                90deg,
                var(--ifx-teal),
                #5bb9b0
            );
        }

        .ifx-detail-card {
            height: 100%;
            background: #ffffff;
            border: 1px solid var(--ifx-line);
            padding: 1rem 1.05rem;
            box-shadow: 0 5px 16px rgba(31, 50, 47, 0.04);
        }

        .ifx-detail-title {
            margin-bottom: 0.72rem;
            color: var(--ifx-ink);
            font-size: 0.95rem;
            font-weight: 730;
        }

        .ifx-detail-grid {
            display: grid;
            grid-template-columns:
                repeat(2, minmax(0, 1fr));
            gap: 0.72rem 1rem;
        }

        .ifx-detail-item {
            min-width: 0;
        }

        .ifx-detail-item.full {
            grid-column: 1 / -1;
        }

        .ifx-detail-label {
            color: var(--ifx-muted);
            font-size: 0.68rem;
            font-weight: 720;
            letter-spacing: 0.035em;
            text-transform: uppercase;
        }

        .ifx-detail-value {
            margin-top: 0.22rem;
            color: var(--ifx-ink);
            font-size: 0.86rem;
            line-height: 1.4;
            font-weight: 600;
            overflow-wrap: anywhere;
            word-break: normal;
        }

        .ifx-sidebar-brand {
            margin-bottom: 0.6rem;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            box-shadow: none !important;
        }

        [data-testid="stForm"] {
            padding: 1rem 1rem 0.25rem 1rem;
        }

        [data-testid="stForm"] textarea {
            min-height: 125px !important;
        }

        [data-testid="stForm"] input,
        [data-testid="stForm"] textarea,
        [data-testid="stForm"] [role="combobox"],
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [role="combobox"] {
            background: #ffffff !important;
            color: var(--ifx-ink) !important;
            -webkit-text-fill-color: var(--ifx-ink) !important;
        }

        [data-testid="stForm"] input:disabled {
            background: #eef1ef !important;
            color: #4e5554 !important;
            -webkit-text-fill-color: #4e5554 !important;
        }

        @media (max-width: 1000px) {
            .ifx-primary-metrics {
                grid-template-columns:
                    repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 720px) {
            .ifx-wafer-metrics,
            .ifx-detail-grid {
                grid-template-columns: 1fr;
            }
        }

        /* V4: review rationale and vertical rhythm */
        .ifx-review-rationale {
            margin-top: 1rem;
            background: #ffffff;
            border: 1px solid var(--ifx-line);
            border-left: 5px solid var(--ifx-teal);
            padding: 1rem 1.05rem;
            box-shadow: 0 5px 16px rgba(31, 50, 47, 0.04);
        }

        .ifx-review-rationale-title {
            color: var(--ifx-ink);
            font-size: 0.96rem;
            font-weight: 740;
            margin-bottom: 0.78rem;
        }

        .ifx-review-rationale-grid {
            display: grid;
            grid-template-columns:
                repeat(2, minmax(0, 1fr));
            gap: 0.82rem 1.3rem;
        }

        .ifx-rationale-item {
            min-width: 0;
        }

        .ifx-rationale-label {
            color: var(--ifx-muted);
            font-size: 0.67rem;
            font-weight: 760;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .ifx-rationale-value {
            margin-top: 0.23rem;
            color: var(--ifx-ink);
            font-size: 0.84rem;
            line-height: 1.46;
            font-weight: 590;
            overflow-wrap: anywhere;
        }

        .ifx-review-checks {
            margin-top: 0.9rem;
            padding-top: 0.8rem;
            border-top: 1px solid var(--ifx-line);
            color: var(--ifx-muted);
            font-size: 0.78rem;
            line-height: 1.58;
        }

        .ifx-evidence-row-gap {
            height: 1.45rem;
        }

        @media (max-width: 760px) {
            .ifx-review-rationale-grid {
                grid-template-columns: 1fr;
            }
        }
</style>
        """,
        unsafe_allow_html=True,
    )

def queue_with_review_status(
    queue: pd.DataFrame,
    latest: pd.DataFrame,
) -> pd.DataFrame:
    reviewed_ids = set(
        latest["input_id"].astype(str)
        if not latest.empty
        else []
    )
    output = queue.copy()
    output["review_status"] = np.where(
        output["input_id"].astype(str).isin(reviewed_ids),
        "REVIEWED",
        "PENDING",
    )
    return output


def filter_queue(
    queue: pd.DataFrame,
    *,
    search_text: str,
    route_filter: str,
    class_filter: str,
    status_filter: str,
    confidence_range: tuple[float, float],
    sort_mode: str,
) -> pd.DataFrame:
    frame = queue.copy()

    if search_text.strip():
        query = search_text.strip().lower()
        frame = frame.loc[
            frame["input_id"].astype(str).str.lower().str.contains(
                query,
                regex=False,
            )
            | frame["predicted_class_name"]
            .astype(str)
            .str.lower()
            .str.contains(query, regex=False)
        ]

    if route_filter != "All routes":
        frame = frame.loc[
            frame["effective_route"].astype(str) == route_filter
        ]

    if class_filter != "All classes":
        frame = frame.loc[
            frame["predicted_class_name"].astype(str) == class_filter
        ]

    if status_filter == "Pending only":
        frame = frame.loc[frame["review_status"] == "PENDING"]
    elif status_filter == "Reviewed only":
        frame = frame.loc[frame["review_status"] == "REVIEWED"]

    lower, upper = confidence_range
    frame = frame.loc[
        frame["confidence"].astype(float).between(
            lower,
            upper,
            inclusive="both",
        )
    ]

    sort_columns: dict[str, tuple[list[str], list[bool]]] = {
        "Operational priority": (
            [
                "review_status",
                "queue_priority",
                "confidence",
                "probability_margin",
                "input_id",
            ],
            [False, True, True, True, True],
        ),
        "Lowest confidence": (
            ["confidence", "probability_margin", "input_id"],
            [True, True, True],
        ),
        "Highest uncertainty": (
            ["normalized_entropy", "confidence", "input_id"],
            [False, True, True],
        ),
        "Predicted class": (
            ["predicted_class_name", "confidence", "input_id"],
            [True, True, True],
        ),
    }
    columns, ascending = sort_columns[sort_mode]
    return frame.sort_values(
        columns,
        ascending=ascending,
    ).reset_index(drop=True)


def compute_explanation(
    case: pd.Series,
    tensors: np.ndarray,
    lookup: dict[str, int],
    model: torch.nn.Module,
    run_device: torch.device,
    step11: Any,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    input_id = str(case["input_id"])
    if input_id not in lookup:
        raise ConsoleError(f"Input is unavailable: {input_id}")

    image = torch.from_numpy(
        tensors[lookup[input_id]].astype(np.float32, copy=False)
    )
    wafer_map = step11.tensor_to_wafer_map(image)
    active_mask = (
        step11.active_wafer_mask(image)
        .cpu()
        .numpy()
        .astype(bool)
    )

    result = step11.calculate_occlusion_sensitivity(
        model=model,
        image=image,
        target_class_id=int(case["predicted_class_id"]),
        patch_size=12,
        stride=6,
        batch_size=64,
        device=run_device,
    )

    explanation = np.where(
        active_mask,
        result.importance_map,
        0.0,
    ).astype(np.float32)

    maximum_drop = float(
        np.max(
            np.maximum(
                result.signed_probability_change,
                0.0,
            )
        )
    )

    return (
        wafer_map,
        explanation,
        float(result.original_probability),
        maximum_drop,
    )


def get_cached_explanation(
    *,
    case: pd.Series,
    tensors: np.ndarray,
    lookup: dict[str, int],
    model: torch.nn.Module,
    run_device: torch.device,
    step11: Any,
    bundle_id: str,
    checkpoint_sha256: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    cache = st.session_state.setdefault(
        "kavach_explanation_cache",
        {},
    )
    cache_key = (
        f"{bundle_id}:{checkpoint_sha256}:"
        f"{str(case['input_id'])}"
    )

    if cache_key not in cache:
        if len(cache) >= 32:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)

        cache[cache_key] = compute_explanation(
            case,
            tensors,
            lookup,
            model,
            run_device,
            step11,
        )

    return cache[cache_key]


def describe_hotspot(
    wafer_map: np.ndarray,
    explanation: np.ndarray,
) -> dict[str, Any]:
    active = wafer_map != 0
    failing = wafer_map == 2
    active_count = int(active.sum())
    failed_count = int(failing.sum())

    if active_count == 0:
        raise ConsoleError("Wafer map contains no active dies.")

    weights = np.where(active, explanation, 0.0)
    total_weight = float(weights.sum())

    coordinates = np.linspace(-1.0, 1.0, wafer_map.shape[0])
    grid_y, grid_x = np.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )

    if total_weight <= 1e-12:
        centroid_x = 0.0
        centroid_y = 0.0
        mean_radius = 0.0
    else:
        centroid_x = float(
            (weights * grid_x).sum() / total_weight
        )
        centroid_y = float(
            (weights * grid_y).sum() / total_weight
        )
        mean_radius = float(
            (
                weights
                * np.sqrt(grid_x**2 + grid_y**2)
            ).sum()
            / total_weight
        )

    if mean_radius <= 0.35:
        radial_band = "centre"
    elif mean_radius <= 0.75:
        radial_band = "mid-radius"
    else:
        radial_band = "edge"

    if abs(centroid_x) <= 0.15 and abs(centroid_y) <= 0.15:
        quadrant = "central"
    elif centroid_y < 0 and centroid_x < 0:
        quadrant = "upper-left"
    elif centroid_y < 0:
        quadrant = "upper-right"
    elif centroid_x < 0:
        quadrant = "lower-left"
    else:
        quadrant = "lower-right"

    active_values = explanation[active]
    threshold = float(
        np.quantile(active_values, 0.80)
    )
    top_region = (explanation >= threshold) & active
    top_count = int(top_region.sum())
    overlap = int((top_region & failing).sum())

    return {
        "hotspot": f"{radial_band} / {quadrant}",
        "failed_die_ratio": failed_count / active_count,
        "top_region_failed_fraction": (
            overlap / top_count if top_count else 0.0
        ),
        "failed_die_coverage": (
            overlap / failed_count if failed_count else 0.0
        ),
    }


def render_explanation(
    wafer_map: np.ndarray,
    explanation: np.ndarray,
) -> plt.Figure:
    wafer_cmap = ListedColormap(
        ["#eef1ef", "#008578", "#d5006d"]
    )
    wafer_norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5],
        wafer_cmap.N,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.4),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#ffffff")

    for axis in axes:
        axis.set_facecolor("#f7f8f6")
        for spine in axis.spines.values():
            spine.set_visible(False)

    axes[0].imshow(
        wafer_map,
        cmap=wafer_cmap,
        norm=wafer_norm,
        interpolation="nearest",
    )
    axes[0].set_title(
        "Wafer state",
        fontsize=11,
        fontweight="semibold",
        color="#18191b",
        pad=10,
    )
    axes[0].axis("off")
    axes[0].legend(
        handles=[
            Patch(color="#008578", label="Passing die"),
            Patch(color="#d5006d", label="Failing die"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    axes[1].imshow(
        wafer_map,
        cmap=wafer_cmap,
        norm=wafer_norm,
        interpolation="nearest",
        alpha=0.50,
    )
    overlay = axes[1].imshow(
        np.ma.masked_where(
            wafer_map == 0,
            explanation,
        ),
        cmap="YlGnBu",
        interpolation="bilinear",
        alpha=0.88,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(
        "Approved good-die occlusion",
        fontsize=11,
        fontweight="semibold",
        color="#18191b",
        pad=10,
    )
    axes[1].axis("off")

    colorbar = figure.colorbar(
        overlay,
        ax=axes[1],
        fraction=0.046,
        pad=0.035,
    )
    colorbar.set_label(
        "Relative model sensitivity",
        fontsize=8,
        color="#5d6367",
    )
    colorbar.ax.tick_params(
        labelsize=7,
        colors="#5d6367",
    )
    colorbar.outline.set_edgecolor("#d9dddc")

    return figure

def render_probability_bars(case: pd.Series) -> None:
    alternatives = [
        (
            1,
            str(case["top_1_class"]),
            float(case["top_1_probability"]),
        ),
        (
            2,
            str(case["top_2_class"]),
            float(case["top_2_probability"]),
        ),
        (
            3,
            str(case["top_3_class"]),
            float(case["top_3_probability"]),
        ),
    ]

    for rank, class_name, probability in alternatives:
        st.markdown(
            (
                '<div class="kavach-probability-row">'
                '<div class="kavach-probability-label">'
                f"<span>{rank}. {html.escape(class_name)}</span>"
                f"<span>{probability:.2%}</span>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.progress(
            float(np.clip(probability, 0.0, 1.0))
        )


def review_guidance(case: pd.Series) -> str:
    if bool(case["monitoring_override_applied"]):
        return (
            "Monitoring has overridden the original route. Automatic "
            "clearance is suspended; complete engineering review before "
            "any disposition."
        )

    if str(case["effective_route"]) == "ABSTAIN":
        return (
            "The frozen policy withheld a decision because the available "
            "model evidence is insufficient. Verify wafer-map quality, "
            "compare the leading alternatives, and escalate when needed."
        )

    return (
        "The prediction is eligible only for engineer review. Use the "
        "approved occlusion evidence as supporting model evidence and "
        "record an independent engineering disposition."
    )


def status_badge(
    text: str,
    *,
    tone: str = "",
) -> str:
    safe_text = html.escape(text)
    safe_tone = html.escape(tone)
    return (
        f'<span class="kavach-badge {safe_tone}">'
        f"{safe_text}</span>"
    )


def render_sidebar(
    *,
    manifest: dict[str, Any],
    queue: pd.DataFrame,
    chain_message: str,
) -> dict[str, Any]:
    with st.sidebar:
        st.markdown(
            """
            <div class="ifx-sidebar-brand">
                <div class="ifx-brand-name">
                    KAVACH <span>AI</span>
                </div>
                <div class="ifx-brand-subtitle">
                    Semiconductor review operations
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        total = len(queue)
        reviewed = int(
            (queue["review_status"] == "REVIEWED").sum()
        )
        progress = reviewed / total if total else 0.0

        st.markdown(
            '<div class="ifx-section-label">Review progress</div>',
            unsafe_allow_html=True,
        )
        st.progress(progress)
        st.caption(
            f"{reviewed:,} / {total:,} reviewed · {progress:.1%}"
        )

        monitoring = html.escape(
            str(manifest["monitoring_status"])
        )
        audit_text = html.escape(chain_message)

        st.markdown(
            (
                '<div class="ifx-compact-status">'
                '<div class="ifx-status-line">'
                '<span>Monitoring</span>'
                f"<strong>{monitoring}</strong>"
                "</div>"
                '<div class="ifx-status-line">'
                '<span>Audit chain</span>'
                f"<strong>{audit_text}</strong>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        with st.expander("Run details", expanded=False):
            st.caption(f"Bundle\n{manifest['bundle_id']}")
            st.caption(
                "Inference\n"
                f"{manifest['source_inference_run_id']}"
            )
            st.caption("Explanation\nGood-die occlusion only")

        with st.expander("Queue filters", expanded=True):
            search_text = st.text_input(
                "Search",
                placeholder="Case ID or predicted class",
            )

            route_column, status_column = st.columns(2)

            with route_column:
                route_filter = st.selectbox(
                    "Route",
                    [
                        "All routes",
                        "ABSTAIN",
                        "ENGINEER_REVIEW",
                    ],
                )

            with status_column:
                status_filter = st.selectbox(
                    "Status",
                    [
                        "Pending only",
                        "All cases",
                        "Reviewed only",
                    ],
                )

            class_filter = st.selectbox(
                "Predicted class",
                [
                    "All classes",
                    *sorted(
                        queue["predicted_class_name"]
                        .astype(str)
                        .unique()
                        .tolist()
                    ),
                ],
            )

        with st.expander(
            "Confidence and sorting",
            expanded=False,
        ):
            confidence_range = st.slider(
                "Confidence range",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.01,
                format="%.2f",
            )

            sort_mode = st.selectbox(
                "Sort queue",
                [
                    "Operational priority",
                    "Lowest confidence",
                    "Highest uncertainty",
                    "Predicted class",
                ],
            )

        st.markdown(
            (
                '<div class="ifx-quick-reference">'
                '<div class="ifx-quick-reference-title">'
                'Engineer quick reference'
                '</div>'
                '<div class="ifx-quick-reference-subtitle">'
                'Wafer-map legend'
                '</div>'
                '<div class="ifx-legend-row">'
                '<span class="ifx-legend-dot pass"></span>'
                '<span>Passing die</span>'
                '</div>'
                '<div class="ifx-legend-row">'
                '<span class="ifx-legend-dot fail"></span>'
                '<span>Failing die</span>'
                '</div>'
                '<div class="ifx-legend-row">'
                '<span class="ifx-legend-dot inactive"></span>'
                '<span>Inactive / background</span>'
                '</div>'
                '<div class="ifx-quick-reference-subtitle">'
                'Routing guide'
                '</div>'
                '<div class="ifx-route-row">'
                '<span class="ifx-route-chip auto">AUTO_ACCEPT</span>'
                '<span>Normal only · ≥94.42% confidence · ≥10% margin</span>'
                '</div>'
                '<div class="ifx-route-row">'
                '<span class="ifx-route-chip review">'
                'ENGINEER_REVIEW'
                '</span>'
                '<span>Predicted defect</span>'
                '</div>'
                '<div class="ifx-route-row">'
                '<span class="ifx-route-chip">ABSTAIN</span>'
                '<span>Insufficient confidence or separation</span>'
                '</div>'
                '<div class="ifx-quick-reference-note">'
                f'<strong>Review queue:</strong> {total:,} cases<br>'
                f'<strong>Reviewed:</strong> {reviewed:,}<br>'
                'An 80% prediction is not auto-clear approval.'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    return {
        "search_text": search_text,
        "route_filter": route_filter,
        "class_filter": class_filter,
        "status_filter": status_filter,
        "confidence_range": confidence_range,
        "sort_mode": sort_mode,
    }

def render_header(
    manifest: dict[str, Any],
    chain_message: str,
) -> None:
    monitoring = str(manifest["monitoring_status"])
    monitoring_tone = (
        "green"
        if monitoring == "GREEN"
        else "amber"
        if monitoring == "AMBER"
        else "red"
    )

    badges = "".join(
        [
            status_badge(
                f"MONITORING {monitoring}",
                tone=monitoring_tone,
            ),
            status_badge(
                "GOOD-DIE OCCLUSION APPROVED",
                tone="green",
            ),
            status_badge(
                f"AUDIT {chain_message}",
                tone="green",
            ),
        ]
    )

    st.markdown(
        (
            '<section class="ifx-hero">'
            '<div class="ifx-hero-copy">'
            '<div class="ifx-eyebrow">'
            "Kavach AI · Semiconductor operations"
            "</div>"
            '<h1 class="ifx-title">'
            "Engineer Review Workbench"
            "</h1>"
            '<div class="ifx-subtitle">'
            "Governed wafer-map decisions with calibrated routing, "
            "drift-aware controls, faithful model-sensitivity evidence, "
            "and an immutable engineering audit trail."
            "</div>"
            f'<div class="ifx-badge-row">{badges}</div>'
            "</div>"
            '<div class="ifx-hero-visual" aria-hidden="true"></div>'
            "</section>"
        ),
        unsafe_allow_html=True,
    )

def render_case_metrics(case: pd.Series) -> None:
    primary_values = [
        (
            "Prediction",
            str(case["predicted_class_name"]),
        ),
        (
            "Confidence",
            f"{float(case['confidence']):.2%}",
        ),
        (
            "Normalized entropy",
            f"{float(case['normalized_entropy']):.3f}",
        ),
        (
            "Probability margin",
            f"{float(case['probability_margin']):.3f}",
        ),
    ]

    cards = "".join(
        (
            '<div class="ifx-primary-card">'
            f'<div class="ifx-primary-label">'
            f'{html.escape(label)}</div>'
            f'<div class="ifx-primary-value">'
            f'{html.escape(value)}</div>'
            "</div>"
        )
        for label, value in primary_values
    )

    review_status = str(case["review_status"])
    review_class = (
        "reviewed"
        if review_status == "REVIEWED"
        else "pending"
    )

    st.markdown(
        (
            f'<div class="ifx-primary-metrics">{cards}</div>'
            '<div class="ifx-case-state-row">'
            f'<span class="ifx-state-chip route">'
            f'ROUTE · {html.escape(str(case["effective_route"]))}'
            "</span>"
            f'<span class="ifx-state-chip {review_class}">'
            f'STATUS · {html.escape(review_status)}'
            "</span>"
            f'<span class="ifx-state-chip">'
            f'INPUT · {html.escape(str(case["input_id"]))}'
            "</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

def render_case_selector(
    filtered: pd.DataFrame,
) -> tuple[pd.Series, list[str]]:
    input_ids = filtered["input_id"].astype(str).tolist()

    active_id = st.session_state.get(
        "kavach_active_case_id",
    )
    if active_id not in input_ids:
        pending = filtered.loc[
            filtered["review_status"] == "PENDING",
            "input_id",
        ].astype(str).tolist()
        active_id = pending[0] if pending else input_ids[0]
        st.session_state["kavach_active_case_id"] = active_id

    label_lookup = {
        str(row.input_id): (
            f"{row.input_id}  ·  {row.effective_route}  ·  "
            f"{row.predicted_class_name}  ·  "
            f"{float(row.confidence):.1%}  ·  {row.review_status}"
        )
        for row in filtered.itertuples(index=False)
    }

    current_index = input_ids.index(active_id)
    selection_column, position_column = st.columns([4.5, 1.0])

    with selection_column:
        chosen_id = st.selectbox(
            "Active review case",
            options=input_ids,
            index=current_index,
            format_func=lambda value: label_lookup[value],
        )
        st.session_state["kavach_active_case_id"] = chosen_id

    current_index = input_ids.index(chosen_id)

    with position_column:
        st.markdown(
            (
                '<div class="kavach-panel" '
                'style="margin-top:1.55rem;text-align:center;">'
                '<div class="kavach-small">Queue position</div>'
                f"<strong>{current_index + 1:,} / {len(input_ids):,}</strong>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    previous_column, next_pending_column, next_column = st.columns(
        [1, 1.25, 1]
    )

    with previous_column:
        if st.button(
            "← Previous",
            disabled=current_index == 0,
            use_container_width=True,
        ):
            st.session_state["kavach_active_case_id"] = input_ids[
                current_index - 1
            ]
            st.rerun()

    pending_ids = filtered.loc[
        filtered["review_status"] == "PENDING",
        "input_id",
    ].astype(str).tolist()
    next_pending_id: str | None = None

    if pending_ids:
        later_pending = [
            value
            for value in input_ids[current_index + 1 :]
            if value in set(pending_ids)
        ]
        next_pending_id = (
            later_pending[0]
            if later_pending
            else pending_ids[0]
        )
        if next_pending_id == chosen_id and len(pending_ids) == 1:
            next_pending_id = None

    with next_pending_column:
        if st.button(
            "Next pending",
            disabled=next_pending_id is None,
            type="primary",
            use_container_width=True,
        ):
            st.session_state["kavach_active_case_id"] = next_pending_id
            st.rerun()

    with next_column:
        if st.button(
            "Next →",
            disabled=current_index == len(input_ids) - 1,
            use_container_width=True,
        ):
            st.session_state["kavach_active_case_id"] = input_ids[
                current_index + 1
            ]
            st.rerun()

    case = filtered.loc[
        filtered["input_id"].astype(str) == chosen_id
    ].iloc[0]
    return case, input_ids


def render_review_workbench(
    *,
    filtered: pd.DataFrame,
    queue: pd.DataFrame,
    latest: pd.DataFrame,
    database_path: Path,
    manifest: dict[str, Any],
    manifest_hash: str,
    tensors: np.ndarray,
    lookup: dict[str, int],
    model: torch.nn.Module,
    run_device: torch.device,
    step11: Any,
) -> None:
    if filtered.empty:
        st.info("No cases match the active filters.")
        return

    case, input_ids = render_case_selector(filtered)
    selected_input_id = str(case["input_id"])

    render_case_metrics(case)

    st.markdown(
        (
            '<div class="kavach-callout kavach-warning">'
            f"{html.escape(review_guidance(case))}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    evidence_column, decision_column = st.columns(
        [1.68, 0.92],
        gap="large",
    )

    with evidence_column:
        st.markdown("## Wafer evidence")

        with st.spinner(
            "Computing approved good-die occlusion evidence..."
        ):
            try:
                (
                    wafer_map,
                    explanation,
                    reproduced_probability,
                    maximum_drop,
                ) = get_cached_explanation(
                    case=case,
                    tensors=tensors,
                    lookup=lookup,
                    model=model,
                    run_device=run_device,
                    step11=step11,
                    bundle_id=str(manifest["bundle_id"]),
                    checkpoint_sha256=str(
                        manifest["model"]["sha256"]
                    ),
                )
            except Exception as exc:
                st.error(
                    f"Explanation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                st.stop()

        if abs(
            reproduced_probability
            - float(case["confidence"])
        ) > 1e-5:
            st.error(
                "Explanation-time probability does not reproduce "
                "the frozen Step 17 result."
            )
            st.stop()

        figure = render_explanation(
            wafer_map,
            explanation,
        )
        st.pyplot(
            figure,
            clear_figure=True,
            use_container_width=True,
        )
        plt.close(figure)

        hotspot = describe_hotspot(
            wafer_map,
            explanation,
        )

        evidence_values = [
            (
                "Maximum probability drop",
                f"{maximum_drop:.4f}",
            ),
            (
                "Attribution hotspot",
                str(hotspot["hotspot"]),
            ),
            (
                "Wafer failing-die ratio",
                f"{hotspot['failed_die_ratio']:.2%}",
            ),
            (
                "Top-region fail overlap",
                f"{hotspot['top_region_failed_fraction']:.2%}",
            ),
        ]

        evidence_cards = "".join(
            (
                '<div class="ifx-wafer-metric">'
                f'<div class="ifx-wafer-metric-label">'
                f'{html.escape(label)}</div>'
                f'<div class="ifx-wafer-metric-value">'
                f'{html.escape(value)}</div>'
                "</div>"
            )
            for label, value in evidence_values
        )

        st.markdown(
            (
                '<div class="ifx-wafer-metrics">'
                f"{evidence_cards}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.caption(
            "Heat intensity is normalized within this wafer. "
            "The absolute maximum target-probability reduction is "
            "reported separately."
        )

        top_two_gap = (
            float(case["top_1_probability"])
            - float(case["top_2_probability"])
        )

        if str(case["monitoring_status"]) == "GREEN":
            monitoring_interpretation = (
                "Monitoring is GREEN. The review route is driven by "
                "case-level model evidence rather than detected drift."
            )
        else:
            monitoring_interpretation = (
                "Monitoring is not GREEN. Review the monitoring "
                "override and drift evidence before disposition."
            )

        if str(case["effective_route"]) == "ABSTAIN":
            route_interpretation = (
                "The frozen policy withheld automatic disposition "
                "because the prediction did not satisfy the required "
                "confidence and class-separation conditions."
            )
        else:
            route_interpretation = (
                "The frozen policy requires an engineer to verify this "
                "predicted defect before operational disposition."
            )

        st.markdown(
            (
                '<div class="ifx-review-rationale">'
                '<div class="ifx-review-rationale-title">'
                "Why this case needs engineering review"
                "</div>"
                '<div class="ifx-review-rationale-grid">'

                '<div class="ifx-rationale-item">'
                '<div class="ifx-rationale-label">'
                "Routing rationale"
                "</div>"
                f'<div class="ifx-rationale-value">'
                f'{html.escape(route_interpretation)}'
                "</div>"
                "</div>"

                '<div class="ifx-rationale-item">'
                '<div class="ifx-rationale-label">'
                "Prediction separation"
                "</div>"
                f'<div class="ifx-rationale-value">'
                f'Top-1 versus top-2 gap: {top_two_gap:.2%}. '
                f'Leading alternative: '
                f'{html.escape(str(case["top_2_class"]))}.'
                "</div>"
                "</div>"

                '<div class="ifx-rationale-item">'
                '<div class="ifx-rationale-label">'
                "Model-sensitivity focus"
                "</div>"
                f'<div class="ifx-rationale-value">'
                f'The approved occlusion evidence is concentrated at '
                f'{html.escape(str(hotspot["hotspot"]))}.'
                "</div>"
                "</div>"

                '<div class="ifx-rationale-item">'
                '<div class="ifx-rationale-label">'
                "Monitoring interpretation"
                "</div>"
                f'<div class="ifx-rationale-value">'
                f'{html.escape(monitoring_interpretation)}'
                "</div>"
                "</div>"

                "</div>"
                '<div class="ifx-review-checks">'
                "<strong>Recommended checks:</strong> verify wafer-map "
                "quality, inspect the highlighted region, compare the "
                "leading competing classes, and record an independent "
                "engineering disposition. The heatmap is model evidence, "
                "not a physical root-cause diagnosis."
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with decision_column:
        st.markdown("## Engineer disposition")

        action_labels = {
            "CONFIRM_MODEL_LABEL": "Confirm model label",
            "OVERRIDE_LABEL": "Override predicted label",
            "ESCALATE_FOR_PROCESS_REVIEW": (
                "Escalate for process review"
            ),
            "MARK_DATA_QUALITY_ISSUE": (
                "Mark data-quality issue"
            ),
        }

        with st.form(
            f"review_form_{selected_input_id}",
            clear_on_submit=False,
        ):
            engineer_id = st.text_input(
                "Engineer ID",
                key="kavach_engineer_id",
                placeholder="e.g., ENG-1042",
            )

            action = st.selectbox(
                "Disposition",
                ACTIONS,
                format_func=lambda value: action_labels[value],
            )

            class_names = [
                str(value)
                for value in manifest["model"]["class_names"]
            ]

            if action == "OVERRIDE_LABEL":
                override_options = [
                    value
                    for value in class_names
                    if value
                    != str(case["predicted_class_name"])
                ]
                selected_class = st.selectbox(
                    "Engineer-selected class",
                    override_options,
                )
            elif action == "CONFIRM_MODEL_LABEL":
                selected_class = str(
                    case["predicted_class_name"]
                )
                st.text_input(
                    "Final class",
                    value=selected_class,
                    disabled=True,
                )
            else:
                selected_class = None

            notes = st.text_area(
                "Engineering notes",
                height=135,
                placeholder=(
                    "Record observed wafer evidence, disposition logic, "
                    "and required follow-up. Do not claim physical root "
                    "cause without factory data."
                ),
            )

            acknowledgement = st.checkbox(
                "I confirm that the heatmap is model-sensitivity "
                "evidence only and that this disposition is based on "
                "independent engineering review."
            )

            advance_after_save = st.checkbox(
                "Open the next pending case after saving",
                value=True,
            )

            submitted = st.form_submit_button(
                "Save immutable review event",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            errors: list[str] = []

            if not engineer_id.strip():
                errors.append("Engineer ID is required.")

            if len(notes.strip()) < 10:
                errors.append(
                    "Engineering notes must contain at least "
                    "10 characters."
                )

            if not acknowledgement:
                errors.append(
                    "The engineering acknowledgement is required."
                )

            if (
                action == "OVERRIDE_LABEL"
                and selected_class
                == str(case["predicted_class_name"])
            ):
                errors.append(
                    "The override class must differ from "
                    "the model prediction."
                )

            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    event_id = append_event(
                        database_path,
                        manifest=manifest,
                        manifest_hash=manifest_hash,
                        case=case,
                        engineer_id=engineer_id,
                        action=action,
                        selected_label=selected_class,
                        notes=notes,
                    )

                    chain_ok, message = verify_chain(
                        database_path
                    )
                    if not chain_ok:
                        st.error(
                            "Review event was written, but audit-chain "
                            f"verification failed: {message}"
                        )
                        st.stop()

                    if advance_after_save:
                        reviewed_ids = set(
                            latest["input_id"]
                            .astype(str)
                            .tolist()
                        )
                        reviewed_ids.add(selected_input_id)

                        pending_after_current = [
                            value
                            for value in input_ids
                            if value not in reviewed_ids
                        ]

                        if pending_after_current:
                            st.session_state[
                                "kavach_active_case_id"
                            ] = pending_after_current[0]

                    st.session_state[
                        "kavach_flash_success"
                    ] = (
                        "Immutable review event saved. "
                        f"Event ID: {event_id}"
                    )
                    st.rerun()

                except Exception as exc:
                    st.error(
                        "Decision could not be appended: "
                        f"{type(exc).__name__}: {exc}"
                    )

    # Decision evidence now spans the full page width.
    st.markdown(
        '<div class="ifx-decision-evidence"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Decision evidence")
    st.caption(
        "Review the model alternatives, monitoring controls, and exact "
        "case lineage after inspecting the wafer evidence."
    )

    alternatives = [
        (
            1,
            str(case["top_1_class"]),
            float(case["top_1_probability"]),
        ),
        (
            2,
            str(case["top_2_class"]),
            float(case["top_2_probability"]),
        ),
        (
            3,
            str(case["top_3_class"]),
            float(case["top_3_probability"]),
        ),
    ]

    alternative_columns = st.columns(3, gap="medium")
    for column, (
        rank,
        class_name,
        probability,
    ) in zip(alternative_columns, alternatives):
        width = float(np.clip(probability, 0.0, 1.0)) * 100.0
        with column:
            st.markdown(
                (
                    '<div class="ifx-alt-card">'
                    f'<div class="ifx-alt-rank">Rank {rank}</div>'
                    '<div class="ifx-alt-header">'
                    f'<div class="ifx-alt-class">'
                    f'{html.escape(class_name)}</div>'
                    f'<div class="ifx-alt-probability">'
                    f'{probability:.2%}</div>'
                    "</div>"
                    '<div class="ifx-native-progress">'
                    f'<div style="width:{width:.3f}%"></div>'
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="ifx-evidence-row-gap"></div>',
        unsafe_allow_html=True,
    )

    context_column, lineage_column = st.columns(
        [1.15, 0.85],
        gap="medium",
    )

    monitoring_reason = html.escape(
        str(case["monitoring_reason"])
    )

    with context_column:
        st.markdown(
            (
                '<div class="ifx-detail-card">'
                '<div class="ifx-detail-title">'
                "Monitoring context"
                "</div>"
                '<div class="ifx-detail-grid">'
                '<div class="ifx-detail-item">'
                '<div class="ifx-detail-label">Status</div>'
                f'<div class="ifx-detail-value">'
                f'{html.escape(str(case["monitoring_status"]))}'
                "</div></div>"
                '<div class="ifx-detail-item">'
                '<div class="ifx-detail-label">'
                "Auto-clear suspended"
                "</div>"
                f'<div class="ifx-detail-value">'
                f'{html.escape(str(bool(case["auto_clearance_suspended"])))}'
                "</div></div>"
                '<div class="ifx-detail-item">'
                '<div class="ifx-detail-label">'
                "Override applied"
                "</div>"
                f'<div class="ifx-detail-value">'
                f'{html.escape(str(bool(case["monitoring_override_applied"])))}'
                "</div></div>"
                '<div class="ifx-detail-item full">'
                '<div class="ifx-detail-label">Reason</div>'
                f'<div class="ifx-detail-value">'
                f"{monitoring_reason}"
                "</div></div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    source_name = Path(str(case["source_file"])).name

    with lineage_column:
        st.markdown(
            (
                '<div class="ifx-detail-card">'
                '<div class="ifx-detail-title">Case lineage</div>'
                '<div class="ifx-detail-grid">'
                '<div class="ifx-detail-item full">'
                '<div class="ifx-detail-label">Input ID</div>'
                f'<div class="ifx-detail-value">'
                f'{html.escape(selected_input_id)}'
                "</div></div>"
                '<div class="ifx-detail-item full">'
                '<div class="ifx-detail-label">Source file</div>'
                f'<div class="ifx-detail-value">'
                f'{html.escape(source_name)}'
                "</div></div>"
                '<div class="ifx-detail-item">'
                '<div class="ifx-detail-label">Source index</div>'
                f'<div class="ifx-detail-value">'
                f'{int(case["source_index"])}'
                "</div></div>"
                '<div class="ifx-detail-item">'
                '<div class="ifx-detail-label">Original route</div>'
                f'<div class="ifx-detail-value">'
                f'{html.escape(str(case["route"]))}'
                "</div></div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    prior = latest.loc[
        latest["input_id"].astype(str) == selected_input_id
    ]

    if not prior.empty:
        current = prior.iloc[0]
        st.markdown(
            (
                '<div class="ifx-detail-card" '
                'style="margin-top:.8rem;">'
                '<div class="ifx-detail-title">'
                "Latest engineer event"
                "</div>"
                '<div class="ifx-detail-grid">'
                '<div class="ifx-detail-item">'
                '<div class="ifx-detail-label">Action</div>'
                f'<div class="ifx-detail-value">'
                f'{html.escape(str(current["action"]))}'
                "</div></div>"
                '<div class="ifx-detail-item">'
                '<div class="ifx-detail-label">Final label</div>'
                f'<div class="ifx-detail-value">'
                f'{html.escape(str(current["selected_label"] or "—"))}'
                "</div></div>"
                '<div class="ifx-detail-item">'
                '<div class="ifx-detail-label">Engineer</div>'
                f'<div class="ifx-detail-value">'
                f'{html.escape(str(current["engineer_id"]))}'
                "</div></div>"
                '<div class="ifx-detail-item">'
                '<div class="ifx-detail-label">UTC time</div>'
                f'<div class="ifx-detail-value">'
                f'{html.escape(str(current["created_at_utc"]))}'
                "</div></div>"
                '<div class="ifx-detail-item full">'
                '<div class="ifx-detail-label">Notes</div>'
                f'<div class="ifx-detail-value">'
                f'{html.escape(str(current["notes"]))}'
                "</div></div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

def _render_infineon_bar_chart(
    counts: pd.Series,
    *,
    title: str,
    horizontal: bool,
) -> None:
    labels = [str(value) for value in counts.index.tolist()]
    values = counts.to_numpy(dtype=float)

    figure, axis = plt.subplots(
        figsize=(7.2, 3.7),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")

    palette = [
        "#008578",
        "#18a095",
        "#5bb9b0",
        "#005b7f",
        "#d5006d",
        "#79c8c0",
        "#7f8c8d",
        "#c4d8d5",
    ]
    colors = [
        palette[index % len(palette)]
        for index in range(len(values))
    ]

    if horizontal:
        positions = np.arange(len(labels))
        bars = axis.barh(
            positions,
            values,
            color=colors,
            height=0.58,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            labels,
            fontsize=9,
            color="#323537",
        )
        axis.invert_yaxis()
        axis.set_xlabel("Cases", color="#5d6367")
        axis.bar_label(
            bars,
            fmt="%.0f",
            padding=4,
            fontsize=8,
            color="#323537",
        )
    else:
        positions = np.arange(len(labels))
        bars = axis.bar(
            positions,
            values,
            color=colors,
            width=0.62,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            labels,
            rotation=35,
            ha="right",
            fontsize=8,
            color="#323537",
        )
        axis.set_ylabel("Cases", color="#5d6367")
        axis.bar_label(
            bars,
            fmt="%.0f",
            padding=3,
            fontsize=8,
            color="#323537",
        )

    axis.set_title(
        title,
        loc="left",
        fontsize=12,
        fontweight="semibold",
        color="#18191b",
        pad=12,
    )
    axis.grid(
        axis="x" if horizontal else "y",
        color="#e5e8e7",
        linewidth=0.8,
    )
    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#d9dddc")
    axis.spines["bottom"].set_color("#d9dddc")
    axis.tick_params(colors="#5d6367")

    st.pyplot(
        figure,
        clear_figure=True,
        use_container_width=True,
    )
    plt.close(figure)

def render_queue_analytics(
    queue: pd.DataFrame,
) -> None:
    total = len(queue)
    reviewed = int(
        (queue["review_status"] == "REVIEWED").sum()
    )
    pending = total - reviewed

    progress_columns = st.columns(4)
    progress_columns[0].metric("Total review queue", f"{total:,}")
    progress_columns[1].metric("Pending", f"{pending:,}")
    progress_columns[2].metric("Reviewed", f"{reviewed:,}")
    progress_columns[3].metric(
        "Completion",
        f"{reviewed / total:.1%}" if total else "0.0%",
    )

    st.progress(reviewed / total if total else 0.0)

    route_column, class_column = st.columns(2)

    route_counts = (
        queue["effective_route"]
        .astype(str)
        .value_counts()
    )
    class_counts = (
        queue["predicted_class_name"]
        .astype(str)
        .value_counts()
    )

    with route_column:
        _render_infineon_bar_chart(
            route_counts,
            title="Queue by effective route",
            horizontal=True,
        )

    with class_column:
        _render_infineon_bar_chart(
            class_counts,
            title="Queue by predicted class",
            horizontal=True,
        )

    st.markdown("### Operational review queue")

    table_columns = [
        "input_id",
        "effective_route",
        "predicted_class_name",
        "confidence",
        "normalized_entropy",
        "probability_margin",
        "monitoring_status",
        "review_status",
    ]
    table = queue[table_columns].copy()
    table["confidence"] = table["confidence"].map(
        lambda value: f"{float(value):.2%}"
    )
    table["normalized_entropy"] = table[
        "normalized_entropy"
    ].map(lambda value: f"{float(value):.3f}")
    table["probability_margin"] = table[
        "probability_margin"
    ].map(lambda value: f"{float(value):.3f}")

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        height=470,
    )

    st.download_button(
        "Download current review queue",
        data=queue.to_csv(index=False).encode("utf-8"),
        file_name="kavach_engineer_review_queue.csv",
        mime="text/csv",
    )

def render_audit_trail(
    *,
    database_path: Path,
    manifest: dict[str, Any],
) -> None:
    chain_ok, chain_message = verify_chain(database_path)
    events = all_events(
        database_path,
        str(manifest["bundle_id"]),
    )

    status_columns = st.columns(3)
    status_columns[0].metric(
        "Audit-chain state",
        "VERIFIED" if chain_ok else "FAILED",
    )
    status_columns[1].metric(
        "Immutable events",
        f"{len(events):,}",
    )
    status_columns[2].metric(
        "Reviewed cases",
        (
            f"{events['input_id'].nunique():,}"
            if not events.empty
            else "0"
        ),
    )

    if not chain_ok:
        st.error(chain_message)
        return

    st.success(chain_message)

    if events.empty:
        st.info(
            "No engineer events have been written to this database."
        )
        return

    display_columns = [
        "sequence_id",
        "created_at_utc",
        "input_id",
        "engineer_id",
        "action",
        "model_label",
        "selected_label",
        "monitoring_status",
        "effective_route",
        "event_id",
        "supersedes_event_id",
    ]
    st.dataframe(
        events[display_columns],
        hide_index=True,
        use_container_width=True,
        height=520,
    )

    st.download_button(
        "Export immutable audit events",
        data=events.to_csv(index=False).encode("utf-8"),
        file_name="kavach_engineer_review_audit_events.csv",
        mime="text/csv",
    )


def render_governance(
    *,
    manifest: dict[str, Any],
    bundle: Path,
    database_path: Path,
) -> None:
    st.markdown("### Frozen artifact lineage")

    lineage = pd.DataFrame(
        [
            {
                "Asset": "Review bundle",
                "Identifier": manifest["bundle_id"],
                "Status": manifest["status"],
            },
            {
                "Asset": "Inference run",
                "Identifier": manifest[
                    "source_inference_run_id"
                ],
                "Status": "COMPLETED",
            },
            {
                "Asset": "Monitoring",
                "Identifier": manifest["monitoring_status"],
                "Status": "VALIDATED",
            },
            {
                "Asset": "Explanation method",
                "Identifier": manifest[
                    "explanation_policy"
                ]["approved_method"],
                "Status": "APPROVED",
            },
        ]
    )
    st.dataframe(
        lineage,
        hide_index=True,
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Integrity controls")
        st.write(
            "- Bundle file checksums are verified before the UI loads."
        )
        st.write(
            "- Model and policy SHA-256 values are verified."
        )
        st.write(
            "- Input-file hashes and wafer counts are verified."
        )
        st.write(
            "- Engineer events are append-only and globally hash-chained."
        )
        st.write(
            "- Corrections are new events; historical events are not edited."
        )

    with right:
        st.markdown("### Operational limitations")
        st.write(
            "- The approved heatmap explains model sensitivity, not "
            "physical semiconductor causality."
        )
        st.write(
            "- WM-811K contains no recipe, equipment, SPC, MES, "
            "maintenance, or yield-history variables."
        )
        st.write(
            "- SQLite is suitable for this governed single-host prototype."
        )
        st.write(
            "- Factory deployment requires authentication, role-based "
            "access, managed transactional storage, backups, and "
            "validated MES/SPC integration."
        )

    st.markdown("### Runtime locations")
    st.code(
        (
            f"Bundle:   {bundle}\n"
            f"Database: {database_path}\n"
            f"Model:    {manifest['model']['path']}\n"
            f"Policy:   {manifest['policy']['path']}"
        ),
        language="text",
    )

    with st.expander("View frozen bundle manifest"):
        st.json(manifest)


def main() -> None:
    st.set_page_config(
        page_title="Kavach AI | Engineer Review",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    cli = parse_cli_args()

    try:
        bundle = resolve_bundle(cli.bundle)
        manifest_path = bundle / "REVIEW_BUNDLE_MANIFEST.json"
        manifest_hash = sha256_file(manifest_path)

        (
            manifest,
            base_queue,
            tensors,
            lookup,
            model,
            run_device,
            step11,
        ) = load_runtime(
            str(bundle),
            manifest_hash,
        )

        database_path = cli.database.expanduser().resolve()
        initialise_database(database_path)

        chain_ok, chain_message = verify_chain(
            database_path
        )
        if not chain_ok:
            st.error(
                f"Audit-chain verification failed: {chain_message}"
            )
            st.stop()

    except Exception as exc:
        st.error(
            "Console preflight failed: "
            f"{type(exc).__name__}: {exc}"
        )
        st.stop()

    latest = latest_events(
        database_path,
        str(manifest["bundle_id"]),
    )
    queue = queue_with_review_status(
        base_queue,
        latest,
    )

    filters = render_sidebar(
        manifest=manifest,
        queue=queue,
        chain_message=chain_message,
    )

    filtered = filter_queue(
        queue,
        **filters,
    )

    render_header(
        manifest,
        chain_message,
    )

    flash_message = st.session_state.pop(
        "kavach_flash_success",
        None,
    )
    if flash_message:
        st.success(flash_message)

    top_metrics = st.columns(5)
    top_metrics[0].metric(
        "Review queue",
        f"{len(queue):,}",
    )
    top_metrics[1].metric(
        "Pending",
        f"{int((queue['review_status'] == 'PENDING').sum()):,}",
    )
    top_metrics[2].metric(
        "Reviewed",
        f"{int((queue['review_status'] == 'REVIEWED').sum()):,}",
    )
    top_metrics[3].metric(
        "ABSTAIN",
        f"{int((queue['effective_route'] == 'ABSTAIN').sum()):,}",
    )
    top_metrics[4].metric(
        "Filtered cases",
        f"{len(filtered):,}",
    )

    review_tab, analytics_tab, audit_tab, governance_tab = st.tabs(
        [
            "Review workbench",
            "Queue analytics",
            "Audit trail",
            "Governance",
        ]
    )

    with review_tab:
        render_review_workbench(
            filtered=filtered,
            queue=queue,
            latest=latest,
            database_path=database_path,
            manifest=manifest,
            manifest_hash=manifest_hash,
            tensors=tensors,
            lookup=lookup,
            model=model,
            run_device=run_device,
            step11=step11,
        )

    with analytics_tab:
        render_queue_analytics(queue)

    with audit_tab:
        render_audit_trail(
            database_path=database_path,
            manifest=manifest,
        )

    with governance_tab:
        render_governance(
            manifest=manifest,
            bundle=bundle,
            database_path=database_path,
        )


if __name__ == "__main__":
    main()
