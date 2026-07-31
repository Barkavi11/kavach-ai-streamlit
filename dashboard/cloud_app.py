#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_DASHBOARD = (
    ROOT / "dashboard" / "base_console.py"
)
DEFAULT_DATABASE = (
    ROOT / ".runtime" / "kavach_engineer_review.sqlite3"
)

REVIEW_ROUTES = frozenset({"ENGINEER_REVIEW", "ABSTAIN"})
ALL_ROUTES = frozenset(
    {"AUTO_ACCEPT", "ENGINEER_REVIEW", "ABSTAIN"}
)

REQUIRED_BASE_FUNCTIONS = (
    "resolve_bundle",
    "sha256_file",
    "load_runtime",
    "initialise_database",
    "verify_chain",
    "latest_events",
    "queue_with_review_status",
    "render_sidebar",
    "filter_queue",
    "render_header",
    "render_review_workbench",
    "render_queue_analytics",
    "render_audit_trail",
    "render_governance",
    "inject_styles",
)


class DecisionLedgerError(RuntimeError):
    """Raised when the governed full-decision view is invalid."""


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--base-dashboard",
        type=Path,
        default=DEFAULT_BASE_DASHBOARD,
        help=(
            "Existing canonical Engineer Review Workbench module."
        ),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Frozen Step 20A review bundle.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="Append-only engineer-review database.",
    )
    arguments, _unknown = parser.parse_known_args(sys.argv[1:])
    return arguments


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DecisionLedgerError(message)


def load_base_dashboard(path: Path) -> ModuleType:
    source = path.expanduser().resolve()
    require(
        source.is_file(),
        f"Canonical dashboard not found: {source}",
    )

    specification = importlib.util.spec_from_file_location(
        "kavach_canonical_review_console",
        source,
    )
    require(
        specification is not None
        and specification.loader is not None,
        f"Unable to import canonical dashboard: {source}",
    )

    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)

    missing = [
        name
        for name in REQUIRED_BASE_FUNCTIONS
        if not hasattr(module, name)
    ]
    require(
        not missing,
        (
            "The selected base dashboard is not the complete elegant "
            f"workbench. Missing functions: {missing}"
        ),
    )
    return module


@st.cache_data(show_spinner=False)
def load_governed_ledger(
    inference_path_text: str,
    inference_sha256: str,
    overlay_path_text: str,
    overlay_sha256: str,
    class_count: int,
) -> pd.DataFrame:
    inference_path = Path(inference_path_text)
    overlay_path = Path(overlay_path_text)

    require(
        inference_path.is_file(),
        f"Inference results not found: {inference_path}",
    )
    require(
        overlay_path.is_file(),
        f"Monitoring routing overlay not found: {overlay_path}",
    )
    require(
        sha256_file(inference_path) == inference_sha256,
        "Frozen inference-results checksum mismatch.",
    )
    require(
        sha256_file(overlay_path) == overlay_sha256,
        "Frozen monitoring-overlay checksum mismatch.",
    )

    results = pd.read_csv(inference_path)
    overlay = pd.read_csv(overlay_path)

    required_results = {
        "input_id",
        "predicted_class_id",
        "predicted_class_name",
        "predicted_is_defect",
        "confidence",
        "entropy",
        "route",
        "route_reason",
        "class_auto_accept_threshold",
        "global_abstain_threshold",
        "top_1_class",
        "top_1_probability",
        "top_2_class",
        "top_2_probability",
        "top_3_class",
        "top_3_probability",
        "source_active_dies",
        "source_failed_dies",
        "source_failed_ratio",
    }
    missing_results = sorted(
        required_results.difference(results.columns)
    )
    require(
        not missing_results,
        (
            "Inference results are missing Decision Ledger fields: "
            f"{missing_results}"
        ),
    )

    required_overlay = {
        "input_id",
        "original_route",
        "monitoring_status",
        "auto_clearance_suspended",
        "effective_route",
        "monitoring_override_applied",
        "monitoring_reason",
    }
    missing_overlay = sorted(
        required_overlay.difference(overlay.columns)
    )
    require(
        not missing_overlay,
        (
            "Monitoring overlay is missing Decision Ledger fields: "
            f"{missing_overlay}"
        ),
    )

    require(
        not results["input_id"].astype(str).duplicated().any(),
        "Inference results contain duplicate input IDs.",
    )
    require(
        not overlay["input_id"].astype(str).duplicated().any(),
        "Monitoring overlay contains duplicate input IDs.",
    )
    require(
        len(results) == len(overlay),
        "Inference results and monitoring overlay counts differ.",
    )

    ledger = results.merge(
        overlay[
            [
                "input_id",
                "original_route",
                "monitoring_status",
                "auto_clearance_suspended",
                "effective_route",
                "monitoring_override_applied",
                "monitoring_reason",
            ]
        ],
        on="input_id",
        how="inner",
        validate="one_to_one",
    )

    require(
        len(ledger) == len(results),
        "Not every inference record aligned to monitoring evidence.",
    )
    require(
        ledger["route"].astype(str).equals(
            ledger["original_route"].astype(str)
        ),
        "Step 17 routes differ from the monitoring overlay.",
    )
    require(
        ledger["effective_route"].astype(str).isin(
            ALL_ROUTES
        ).all(),
        "Decision Ledger contains an unsupported effective route.",
    )

    ledger["normalized_entropy"] = (
        ledger["entropy"].astype(float)
        / np.log(class_count)
    )

    if "predicted_vs_best_other_margin" in ledger.columns:
        ledger["probability_margin"] = ledger[
            "predicted_vs_best_other_margin"
        ].astype(float)
    else:
        ledger["probability_margin"] = (
            ledger["top_1_probability"].astype(float)
            - ledger["top_2_probability"].astype(float)
        )

    ledger["predicted_class_name"] = (
        ledger["predicted_class_name"].astype(str)
    )
    ledger["effective_route"] = (
        ledger["effective_route"].astype(str)
    )
    ledger["input_id"] = ledger["input_id"].astype(str)

    route_order = {
        "AUTO_ACCEPT": 0,
        "ABSTAIN": 1,
        "ENGINEER_REVIEW": 2,
    }
    ledger["_route_order"] = (
        ledger["effective_route"]
        .map(route_order)
        .fillna(99)
        .astype(int)
    )

    return ledger.sort_values(
        [
            "_route_order",
            "predicted_class_name",
            "confidence",
            "input_id",
        ],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def attach_decision_status(
    ledger: pd.DataFrame,
    latest: pd.DataFrame,
) -> pd.DataFrame:
    reviewed_ids = set(
        latest["input_id"].astype(str)
        if not latest.empty
        else []
    )

    output = ledger.copy()
    output["decision_status"] = np.select(
        [
            output["effective_route"].eq("AUTO_ACCEPT"),
            output["input_id"].isin(reviewed_ids),
        ],
        [
            "AUTO-CLEARED",
            "REVIEWED",
        ],
        default="PENDING REVIEW",
    )
    return output


def validate_ledger_contract(
    ledger: pd.DataFrame,
    review_queue: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    require(
        len(ledger)
        == int(manifest["source_input_wafer_count"]),
        "Decision Ledger count differs from the frozen input count.",
    )

    ledger_review_ids = set(
        ledger.loc[
            ledger["effective_route"].isin(REVIEW_ROUTES),
            "input_id",
        ].astype(str)
    )
    queue_ids = set(
        review_queue["input_id"].astype(str)
    )

    require(
        ledger_review_ids == queue_ids,
        (
            "The full Decision Ledger and the frozen review queue "
            "do not contain the same review-required cases."
        ),
    )
    require(
        len(queue_ids) == int(manifest["review_queue_records"]),
        "Review queue count differs from the frozen manifest.",
    )


def render_system_header(
    base: ModuleType,
    manifest: dict[str, Any],
    chain_message: str,
) -> None:
    monitoring = str(manifest["monitoring_status"])
    tone = (
        "green"
        if monitoring == "GREEN"
        else "amber"
        if monitoring == "AMBER"
        else "red"
    )

    badges = "".join(
        [
            base.status_badge(
                f"MONITORING {monitoring}",
                tone=tone,
            ),
            base.status_badge(
                "GOOD-DIE OCCLUSION APPROVED",
                tone="green",
            ),
            base.status_badge(
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
            "Governed wafer-map decisions with complete batch "
            "traceability, calibrated routing, drift-aware controls, "
            "faithful model-sensitivity evidence, and an immutable "
            "engineering audit trail."
            "</div>"
            f'<div class="ifx-badge-row">{badges}</div>'
            "</div>"
            '<div class="ifx-hero-visual" aria-hidden="true"></div>'
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def inject_ledger_styles() -> None:
    st.markdown(
        """
        <style>
        .ledger-section-heading {
            margin: 0.25rem 0 0.15rem;
            color: #18191b;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .ledger-section-caption {
            margin-bottom: 1rem;
            color: #5d6367;
            font-size: 0.83rem;
            line-height: 1.45;
        }

        .ledger-status {
            margin-bottom: 0.85rem;
            padding: 0.85rem 0.95rem;
            border-left: 4px solid #008578;
            background: #e6f4f2;
            color: #124d48;
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1.45;
        }

        .ledger-status.review {
            border-left-color: #e28a00;
            background: #fff8e8;
            color: #6e4a08;
        }

        .ledger-status.reviewed {
            border-left-color: #005b7f;
            background: #eaf4f8;
            color: #164c61;
        }

        .ledger-readonly {
            margin-top: 0.75rem;
            color: #5d6367;
            font-size: 0.74rem;
            line-height: 1.45;
        }

        .ledger-kv {
            display: grid;
            grid-template-columns: minmax(7.8rem, 0.9fr) 1.2fr;
            gap: 0.48rem 0.75rem;
            padding: 0.9rem;
            border: 1px solid #d9dddc;
            background: #ffffff;
            font-size: 0.79rem;
        }

        .ledger-kv .label {
            color: #5d6367;
        }

        .ledger-kv .value {
            color: #18191b;
            font-weight: 600;
            overflow-wrap: anywhere;
        }

        .ledger-reason {
            margin-top: 0.75rem;
            padding: 0.75rem 0.85rem;
            border: 1px solid #d9dddc;
            background: #f9faf8;
            color: #44494b;
            font-size: 0.77rem;
            line-height: 1.48;
        }

        .upload-intro {
            margin-bottom: 1rem;
            padding: 0.95rem 1rem;
            border-left: 4px solid #008578;
            background: #eef8f6;
            color: #2f3d3b;
            font-size: 0.80rem;
            line-height: 1.52;
        }

        .upload-warning {
            margin: 0.75rem 0;
            padding: 0.78rem 0.88rem;
            border-left: 4px solid #e28a00;
            background: #fff8e8;
            color: #62480e;
            font-size: 0.76rem;
            line-height: 1.48;
        }

        .upload-result-banner {
            margin-bottom: 0.85rem;
            padding: 0.90rem 1rem;
            border-left: 4px solid #d5006d;
            background: #fff1f7;
            color: #6f1748;
            font-size: 0.81rem;
            font-weight: 750;
            line-height: 1.48;
        }

        .upload-result-banner.normal {
            border-left-color: #008578;
            background: #e9f7f4;
            color: #155f56;
        }

        .upload-metadata {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem 0.8rem;
            margin-top: 0.75rem;
            padding: 0.85rem;
            border: 1px solid #d9dddc;
            background: #ffffff;
            font-size: 0.76rem;
        }

        .upload-metadata .label {
            color: #5d6367;
        }

        .upload-metadata .value {
            color: #18191b;
            font-weight: 650;
            overflow-wrap: anywhere;
        }

        /*
         * Compact the three sidebar control groups. The structural
         * divider has been removed in the compact base dashboard.
         */
        section[data-testid="stSidebar"]
        div[data-testid="stExpander"] {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        section[data-testid="stSidebar"]
        div[data-testid="stElementContainer"]:has(
            div[data-testid="stExpander"]
        ) {
            margin-top: 0 !important;
            margin-bottom: 0.20rem !important;
        }

        section[data-testid="stSidebar"]
        div[data-testid="stExpander"] details > summary {
            min-height: 2.50rem;
            padding-top: 0.42rem;
            padding-bottom: 0.42rem;
        }

        section[data-testid="stSidebar"]
        div[data-testid="stExpander"] details[open] > summary {
            margin-bottom: 0.18rem;
        }

        @media (max-width: 1100px) {
            .ledger-kv {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_wafer_map(
    wafer_map: np.ndarray,
    title: str,
) -> plt.Figure:
    wafer_cmap = ListedColormap(
        ["#eef1ef", "#008578", "#d5006d"]
    )
    wafer_norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5],
        wafer_cmap.N,
    )

    figure, axis = plt.subplots(
        figsize=(4.1, 4.1),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#f7f8f6")
    axis.imshow(
        wafer_map,
        cmap=wafer_cmap,
        norm=wafer_norm,
        interpolation="nearest",
    )
    axis.set_title(
        title,
        fontsize=11,
        fontweight="semibold",
        color="#18191b",
        pad=10,
    )
    axis.axis("off")
    return figure


def case_wafer_map(
    case: pd.Series,
    tensors: np.ndarray,
    lookup: dict[str, int],
    step11: Any,
) -> np.ndarray:
    input_id = str(case["input_id"])
    require(
        input_id in lookup,
        f"Tensor is unavailable for Decision Ledger case {input_id}.",
    )
    image = torch.from_numpy(
        tensors[lookup[input_id]].astype(
            np.float32,
            copy=False,
        )
    )
    return step11.tensor_to_wafer_map(image)


def filter_ledger(
    ledger: pd.DataFrame,
    *,
    search_text: str,
    scope: str,
    route_filter: str,
    class_filter: str,
    status_filter: str,
    confidence_range: tuple[float, float],
) -> pd.DataFrame:
    frame = ledger.copy()

    if scope == "Normal cases":
        frame = frame.loc[
            frame["predicted_class_name"].eq("Normal")
        ]
    elif scope == "Review-required cases":
        frame = frame.loc[
            frame["effective_route"].isin(REVIEW_ROUTES)
        ]

    if search_text.strip():
        query = search_text.strip().lower()
        frame = frame.loc[
            frame["input_id"]
            .str.lower()
            .str.contains(query, regex=False)
            | frame["predicted_class_name"]
            .str.lower()
            .str.contains(query, regex=False)
        ]

    if route_filter != "All routes":
        frame = frame.loc[
            frame["effective_route"].eq(route_filter)
        ]

    if class_filter != "All classes":
        frame = frame.loc[
            frame["predicted_class_name"].eq(class_filter)
        ]

    if status_filter != "All statuses":
        frame = frame.loc[
            frame["decision_status"].eq(status_filter)
        ]

    lower, upper = confidence_range
    frame = frame.loc[
        frame["confidence"].astype(float).between(
            lower,
            upper,
            inclusive="both",
        )
    ]

    return frame.reset_index(drop=True)


def probability_rows(case: pd.Series) -> list[tuple[str, float]]:
    return [
        (
            str(case["top_1_class"]),
            float(case["top_1_probability"]),
        ),
        (
            str(case["top_2_class"]),
            float(case["top_2_probability"]),
        ),
        (
            str(case["top_3_class"]),
            float(case["top_3_probability"]),
        ),
    ]


def optional_percentage(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not applicable"
    return f"{float(value):.2%}"


def render_ledger_case(
    case: pd.Series,
    *,
    tensors: np.ndarray,
    lookup: dict[str, int],
    step11: Any,
) -> None:
    route = str(case["effective_route"])
    status = str(case["decision_status"])

    if route == "AUTO_ACCEPT":
        status_class = ""
        status_title = "AUTO-CLEARED NORMAL"
        status_text = (
            "The frozen policy cleared this Normal prediction. "
            "No engineer disposition is required."
        )
    elif status == "REVIEWED":
        status_class = " reviewed"
        status_title = "ENGINEERING REVIEW COMPLETED"
        status_text = (
            "This case has a recorded engineer event in the "
            "append-only audit trail."
        )
    else:
        status_class = " review"
        status_title = "ENGINEERING REVIEW REQUIRED"
        status_text = (
            "The frozen routing policy withheld automatic clearance. "
            "Complete the disposition in Review workbench."
        )

    st.markdown(
        (
            f'<div class="ledger-status{status_class}">'
            f"{html.escape(status_title)}<br>"
            f'<span style="font-weight:500">'
            f"{html.escape(status_text)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    wafer_map = case_wafer_map(
        case,
        tensors,
        lookup,
        step11,
    )
    figure = render_wafer_map(
        wafer_map,
        title=str(case["input_id"]),
    )
    st.pyplot(
        figure,
        clear_figure=True,
        use_container_width=True,
    )
    plt.close(figure)

    threshold = optional_percentage(
        case.get("class_auto_accept_threshold")
    )

    key_values = [
        ("Predicted class", str(case["predicted_class_name"])),
        ("Confidence", f"{float(case['confidence']):.2%}"),
        ("Effective route", route),
        ("Decision status", status),
        ("Class threshold", threshold),
        (
            "Probability margin",
            f"{float(case['probability_margin']):.2%}",
        ),
        (
            "Normalized entropy",
            f"{float(case['normalized_entropy']):.3f}",
        ),
        (
            "Failed-die ratio",
            f"{float(case['source_failed_ratio']):.2%}",
        ),
        (
            "Active / failed dies",
            (
                f"{int(case['source_active_dies']):,} / "
                f"{int(case['source_failed_dies']):,}"
            ),
        ),
        ("Monitoring", str(case["monitoring_status"])),
    ]

    key_value_html = "".join(
        (
            f'<div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div>'
        )
        for label, value in key_values
    )
    st.markdown(
        f'<div class="ledger-kv">{key_value_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Leading model probabilities")
    for rank, (class_name, probability) in enumerate(
        probability_rows(case),
        start=1,
    ):
        left, right = st.columns([3.4, 1.0])
        with left:
            st.caption(f"{rank}. {class_name}")
            st.progress(
                float(np.clip(probability, 0.0, 1.0))
            )
        with right:
            st.caption(f"{probability:.2%}")

    reason = str(case["route_reason"])
    monitoring_reason = str(case["monitoring_reason"])
    st.markdown(
        (
            '<div class="ledger-reason">'
            f"<strong>Policy rationale:</strong> "
            f"{html.escape(reason)}<br><br>"
            f"<strong>Monitoring context:</strong> "
            f"{html.escape(monitoring_reason)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="ledger-readonly">'
            "Decision Ledger is read-only. Auto-cleared cases do not "
            "create engineer-review events. Review-required cases must "
            "be dispositioned in Review workbench."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_decision_ledger(
    ledger: pd.DataFrame,
    *,
    tensors: np.ndarray,
    lookup: dict[str, int],
    step11: Any,
) -> None:
    total = len(ledger)
    predicted_normal = int(
        ledger["predicted_class_name"].eq("Normal").sum()
    )
    auto_cleared_normal = int(
        (
            ledger["predicted_class_name"].eq("Normal")
            & ledger["effective_route"].eq("AUTO_ACCEPT")
        ).sum()
    )
    withheld_normal = predicted_normal - auto_cleared_normal
    review_required = int(
        ledger["effective_route"].isin(REVIEW_ROUTES).sum()
    )

    st.markdown(
        '<div class="ledger-section-heading">Decision Ledger</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="ledger-section-caption">'
            "Complete read-only traceability for the frozen inference batch. "
            "Normal cases are shown by default; switch the scope to inspect "
            "all routes and predicted classes."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    metrics = st.columns(5, gap="small")
    metrics[0].metric("All decisions", f"{total:,}")
    metrics[1].metric(
        "Predicted Normal",
        f"{predicted_normal:,}",
    )
    metrics[2].metric(
        "Auto-cleared Normal",
        f"{auto_cleared_normal:,}",
    )
    metrics[3].metric(
        "Withheld Normal",
        f"{withheld_normal:,}",
    )
    metrics[4].metric(
        "Requires review",
        f"{review_required:,}",
    )

    filter_columns = st.columns(
        [1.35, 1.05, 1.0, 1.0, 1.0],
        gap="small",
    )
    with filter_columns[0]:
        search_text = st.text_input(
            "Search decisions",
            placeholder="Case ID or class",
            key="ledger_search",
        )
    with filter_columns[1]:
        scope = st.selectbox(
            "Scope",
            [
                "Normal cases",
                "All decisions",
                "Review-required cases",
            ],
            key="ledger_scope",
        )
    with filter_columns[2]:
        route_filter = st.selectbox(
            "Route",
            [
                "All routes",
                "AUTO_ACCEPT",
                "ENGINEER_REVIEW",
                "ABSTAIN",
            ],
            key="ledger_route",
        )
    with filter_columns[3]:
        class_filter = st.selectbox(
            "Predicted class",
            [
                "All classes",
                *sorted(
                    ledger["predicted_class_name"]
                    .unique()
                    .tolist()
                ),
            ],
            key="ledger_class",
        )
    with filter_columns[4]:
        status_filter = st.selectbox(
            "Status",
            [
                "All statuses",
                "AUTO-CLEARED",
                "PENDING REVIEW",
                "REVIEWED",
            ],
            key="ledger_status",
        )

    with st.expander(
        "Confidence filter",
        expanded=False,
    ):
        confidence_range = st.slider(
            "Prediction confidence",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
            format="%.2f",
            key="ledger_confidence",
        )

    filtered = filter_ledger(
        ledger,
        search_text=search_text,
        scope=scope,
        route_filter=route_filter,
        class_filter=class_filter,
        status_filter=status_filter,
        confidence_range=confidence_range,
    )

    st.caption(
        f"Showing {len(filtered):,} of {len(ledger):,} decisions."
    )

    if filtered.empty:
        st.info("No decisions match the selected filters.")
        return

    table_column, inspector_column = st.columns(
        [1.42, 0.88],
        gap="large",
    )

    with table_column:
        table = filtered[
            [
                "input_id",
                "effective_route",
                "decision_status",
                "predicted_class_name",
                "confidence",
                "probability_margin",
                "source_failed_ratio",
                "monitoring_status",
            ]
        ].rename(
            columns={
                "input_id": "Case ID",
                "effective_route": "Route",
                "decision_status": "Status",
                "predicted_class_name": "Prediction",
                "confidence": "Confidence",
                "probability_margin": "Margin",
                "source_failed_ratio": "Failed-die ratio",
                "monitoring_status": "Monitoring",
            }
        )

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            height=700,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
                "Margin": st.column_config.NumberColumn(
                    "Margin",
                    format="%.3f",
                ),
                "Failed-die ratio": st.column_config.NumberColumn(
                    "Failed-die ratio",
                    format="%.3f",
                ),
            },
        )

    with inspector_column:
        labels: list[str] = []
        label_to_id: dict[str, str] = {}
        for row in filtered.itertuples(index=False):
            label = (
                f"{row.input_id} · {row.effective_route} · "
                f"{row.predicted_class_name} · "
                f"{float(row.confidence):.1%}"
            )
            labels.append(label)
            label_to_id[label] = str(row.input_id)

        chosen_label = st.selectbox(
            "Inspect decision",
            labels,
            key="ledger_selected_case",
        )
        selected_id = label_to_id[chosen_label]
        case = filtered.loc[
            filtered["input_id"].eq(selected_id)
        ].iloc[0]

        render_ledger_case(
            case,
            tensors=tensors,
            lookup=lookup,
            step11=step11,
        )



PASS_RGB = np.array([0, 133, 120], dtype=np.float32)
FAIL_RGB = np.array([213, 0, 109], dtype=np.float32)
BACKGROUND_RGB = np.array([238, 241, 239], dtype=np.float32)


def validate_raw_wafer_map(
    array: np.ndarray,
    *,
    source_name: str,
) -> np.ndarray:
    wafer = np.asarray(array)
    wafer = np.squeeze(wafer)

    require(
        wafer.ndim == 2,
        (
            f"{source_name} must contain one two-dimensional wafer map; "
            f"received shape {wafer.shape}."
        ),
    )
    require(
        8 <= wafer.shape[0] <= 2048
        and 8 <= wafer.shape[1] <= 2048,
        (
            "Wafer-map dimensions must be between 8 and 2,048 pixels "
            f"per side; received {wafer.shape}."
        ),
    )
    require(
        np.issubdtype(wafer.dtype, np.number),
        "Wafer-map arrays must be numeric.",
    )
    require(
        np.isfinite(wafer).all(),
        "Wafer-map arrays cannot contain NaN or infinite values.",
    )

    rounded = np.rint(wafer).astype(np.int16)
    require(
        np.allclose(
            wafer.astype(np.float64),
            rounded.astype(np.float64),
            atol=1e-6,
        ),
        "Wafer-map values must be discrete 0, 1, or 2.",
    )

    unique_values = set(np.unique(rounded).tolist())
    require(
        unique_values.issubset({0, 1, 2}),
        (
            "Unsupported wafer-map encoding. Expected "
            "0=inactive, 1=passing, 2=failing; found "
            f"{sorted(unique_values)}."
        ),
    )
    require(
        int(np.count_nonzero(rounded)) >= 25,
        "The uploaded wafer map contains too few active die pixels.",
    )

    return rounded.astype(np.uint8)


def parse_palette_png(
    payload: bytes,
    *,
    source_name: str,
) -> np.ndarray:
    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise DecisionLedgerError(
            f"{source_name} is not a readable PNG image."
        ) from exc

    rgb = np.asarray(image, dtype=np.float32)
    require(
        rgb.ndim == 3 and rgb.shape[2] == 3,
        "Uploaded PNG must be an RGB wafer-map image.",
    )
    require(
        rgb.shape[0] <= 4096 and rgb.shape[1] <= 4096,
        "Uploaded PNG is too large; maximum supported size is 4,096×4,096.",
    )

    pass_distance = np.linalg.norm(
        rgb - PASS_RGB.reshape(1, 1, 3),
        axis=2,
    )
    fail_distance = np.linalg.norm(
        rgb - FAIL_RGB.reshape(1, 1, 3),
        axis=2,
    )
    active_distance = np.minimum(
        pass_distance,
        fail_distance,
    )
    active_candidate = active_distance <= 105.0

    require(
        int(active_candidate.sum()) >= 25,
        (
            "No canonical teal/magenta wafer map was detected. "
            "Upload a clean Kavach AI palette PNG or a raw NPY/NPZ map."
        ),
    )

    row_threshold = max(
        2,
        int(np.ceil(rgb.shape[1] * 0.003)),
    )
    column_threshold = max(
        2,
        int(np.ceil(rgb.shape[0] * 0.003)),
    )
    active_rows = np.where(
        active_candidate.sum(axis=1) >= row_threshold
    )[0]
    active_columns = np.where(
        active_candidate.sum(axis=0) >= column_threshold
    )[0]

    require(
        active_rows.size > 0 and active_columns.size > 0,
        "Unable to isolate the wafer region from the PNG.",
    )

    top = int(active_rows.min())
    bottom = int(active_rows.max()) + 1
    left = int(active_columns.min())
    right = int(active_columns.max()) + 1
    crop = rgb[top:bottom, left:right]

    palette = np.stack(
        [
            BACKGROUND_RGB,
            PASS_RGB,
            FAIL_RGB,
        ],
        axis=0,
    )
    distances = np.linalg.norm(
        crop[:, :, None, :] - palette[None, None, :, :],
        axis=3,
    )
    nearest = np.argmin(distances, axis=2)
    minimum_distance = np.min(distances, axis=2)

    wafer = nearest.astype(np.uint8)
    wafer[minimum_distance > 125.0] = 0

    active = wafer > 0
    require(
        int(active.sum()) >= 25,
        "The parsed PNG contains too few active die pixels.",
    )

    rows = np.where(active.any(axis=1))[0]
    columns = np.where(active.any(axis=0))[0]
    wafer = wafer[
        int(rows.min()): int(rows.max()) + 1,
        int(columns.min()): int(columns.max()) + 1,
    ]

    return validate_raw_wafer_map(
        wafer,
        source_name=source_name,
    )


def parse_uploaded_wafer(
    *,
    uploaded_name: str,
    payload: bytes,
    npz_key: str | None,
) -> np.ndarray:
    suffix = Path(uploaded_name).suffix.lower()

    require(
        len(payload) <= 25 * 1024 * 1024,
        "Upload exceeds the 25 MB safety limit.",
    )

    if suffix == ".png":
        return parse_palette_png(
            payload,
            source_name=uploaded_name,
        )

    if suffix == ".npy":
        array = np.load(
            BytesIO(payload),
            allow_pickle=False,
        )
        return validate_raw_wafer_map(
            array,
            source_name=uploaded_name,
        )

    if suffix == ".npz":
        with np.load(
            BytesIO(payload),
            allow_pickle=False,
        ) as archive:
            keys = list(archive.files)
            require(
                keys,
                "The NPZ archive does not contain any arrays.",
            )
            selected_key = npz_key or keys[0]
            require(
                selected_key in keys,
                f"NPZ key not found: {selected_key}",
            )
            array = archive[selected_key]
        return validate_raw_wafer_map(
            array,
            source_name=f"{uploaded_name}:{selected_key}",
        )

    raise DecisionLedgerError(
        "Unsupported upload type. Use PNG, NPY, or NPZ."
    )


def classify_uploaded_wafer(
    *,
    base: ModuleType,
    raw_wafer: np.ndarray,
    model: torch.nn.Module,
    run_device: torch.device,
    step11: Any,
    class_names: list[str],
) -> dict[str, Any]:
    step04 = sys.modules.get("kavach_step04_console")
    if step04 is None:
        step04 = base.load_module(
            base.STEP04,
            "kavach_step04_upload_classifier",
        )

    step17 = sys.modules.get("kavach_step17_console")
    if step17 is None:
        step17 = base.load_module(
            base.STEP17,
            "kavach_step17_upload_classifier",
        )

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".npy",
            delete=False,
        ) as temporary_file:
            np.save(
                temporary_file,
                raw_wafer,
                allow_pickle=False,
            )
            temporary_path = temporary_file.name

        wafers = step17.load_input_files(
            [Path(temporary_path)],
            npz_key=None,
            maximum_wafers=1,
        )
        require(
            len(wafers) == 1,
            "Upload preprocessing did not produce exactly one wafer.",
        )

        tensors, diagnostics = step17.preprocess_wafers(
            wafers,
            step04=step04,
            maximum_side=512,
        )
        require(
            len(tensors) == 1,
            "Upload preprocessing did not produce exactly one tensor.",
        )

        image_tensor = torch.from_numpy(
            tensors[0].astype(
                np.float32,
                copy=False,
            )
        )
        batch = image_tensor.unsqueeze(0).to(run_device)

        with torch.inference_mode():
            logits = model(batch)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0].detach().cpu().numpy()

        require(
            probabilities.shape[0] == len(class_names),
            "Model output and class-name counts differ.",
        )
        require(
            np.isfinite(probabilities).all(),
            "Model returned non-finite probabilities.",
        )

        ranking = np.argsort(probabilities)[::-1]
        top_indices = ranking[:3]
        top_predictions = [
            {
                "class_name": class_names[int(index)],
                "probability": float(probabilities[int(index)]),
            }
            for index in top_indices
        ]

        predicted_index = int(top_indices[0])
        confidence = float(probabilities[predicted_index])
        second_probability = float(
            probabilities[int(top_indices[1])]
        )
        margin = confidence - second_probability
        predicted_class = class_names[predicted_index]

        model_wafer = step11.tensor_to_wafer_map(
            image_tensor
        )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "margin": margin,
            "top_predictions": top_predictions,
            "wafer_map": model_wafer,
            "raw_shape": list(raw_wafer.shape),
            "model_shape": list(image_tensor.shape),
            "active_dies": int(np.count_nonzero(raw_wafer)),
            "failed_dies": int(np.count_nonzero(raw_wafer == 2)),
            "diagnostics": diagnostics,
        }
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(
                missing_ok=True,
            )


def render_upload_classifier(
    *,
    base: ModuleType,
    manifest: dict[str, Any],
    model: torch.nn.Module,
    run_device: torch.device,
    step11: Any,
) -> None:
    st.markdown(
        '<div class="ledger-section-heading">Upload classifier</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="upload-intro">'
            "<strong>Purpose:</strong> classify one wafer-map pattern "
            "with the frozen Kavach AI CNN. Accepted inputs are a raw "
            "two-dimensional NPY/NPZ map encoded as "
            "<code>0=inactive, 1=passing, 2=failing</code>, or a clean "
            "Kavach AI palette PNG."
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="upload-warning">'
            "<strong>Important:</strong> this model does not classify "
            "microscope photographs, SEM images, equipment photos, or "
            "general screenshots. Uploaded inputs are always treated as "
            "engineering-review candidates and are never auto-cleared."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    input_column, guidance_column = st.columns(
        [1.25, 0.75],
        gap="large",
    )

    with input_column:
        uploaded = st.file_uploader(
            "Upload one wafer map",
            type=["png", "npy", "npz"],
            accept_multiple_files=False,
            help=(
                "Use a raw NPY/NPZ map whenever possible. PNG support "
                "expects the Kavach AI teal/magenta wafer palette."
            ),
            key="upload_classifier_file",
        )

        npz_key: str | None = None
        payload: bytes | None = None

        if uploaded is not None:
            payload = uploaded.getvalue()
            if Path(uploaded.name).suffix.lower() == ".npz":
                try:
                    with np.load(
                        BytesIO(payload),
                        allow_pickle=False,
                    ) as archive:
                        available_keys = list(archive.files)
                except Exception as exc:
                    st.error(
                        "Unable to inspect the NPZ archive: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    available_keys = []

                if available_keys:
                    npz_key = st.selectbox(
                        "NPZ array",
                        available_keys,
                        key="upload_classifier_npz_key",
                    )

        acknowledgement = st.checkbox(
            (
                "I confirm this is a wafer-map input and understand "
                "that the result requires engineer review."
            ),
            key="upload_classifier_ack",
        )

        run_clicked = st.button(
            "Classify wafer map",
            type="primary",
            use_container_width=True,
            disabled=(
                uploaded is None
                or not acknowledgement
            ),
            key="upload_classifier_run",
        )

    with guidance_column:
        st.markdown("#### Supported input")
        st.caption(
            "Best: `.npy` or `.npz` containing one 2D wafer map."
        )
        st.caption(
            "Image option: clean `.png` using teal passing dies, "
            "magenta failing dies, and a light background."
        )
        st.markdown("#### Output")
        st.caption(
            "Predicted class, confidence, top-three probabilities, "
            "probability margin, and the model-view wafer map."
        )
        st.markdown("#### Classes")
        st.caption(
            "Normal, Center, Donut, Edge-Loc, Edge-Ring, Loc, "
            "Near-full, Random, Scratch."
        )

    if run_clicked and uploaded is not None and payload is not None:
        try:
            raw_wafer = parse_uploaded_wafer(
                uploaded_name=uploaded.name,
                payload=payload,
                npz_key=npz_key,
            )
            result = classify_uploaded_wafer(
                base=base,
                raw_wafer=raw_wafer,
                model=model,
                run_device=run_device,
                step11=step11,
                class_names=[
                    str(value)
                    for value in manifest["model"]["class_names"]
                ],
            )
            st.session_state[
                "upload_classifier_result"
            ] = result
            st.session_state[
                "upload_classifier_source_name"
            ] = uploaded.name
        except Exception as exc:
            st.session_state.pop(
                "upload_classifier_result",
                None,
            )
            st.error(
                "Upload classification failed: "
                f"{type(exc).__name__}: {exc}"
            )

    result = st.session_state.get(
        "upload_classifier_result"
    )
    if result is None:
        return

    predicted_class = str(result["predicted_class"])
    is_normal = predicted_class == "Normal"
    banner_class = " normal" if is_normal else ""
    st.markdown(
        (
            f'<div class="upload-result-banner{banner_class}">'
            f"Predicted class: {html.escape(predicted_class)} · "
            f"Confidence: {float(result['confidence']):.2%} · "
            f"Top-1/Top-2 margin: {float(result['margin']):.2%}<br>"
            '<span style="font-weight:500">'
            "Operational handling: ENGINEER_REVIEW — uploaded inputs "
            "are not part of the frozen automatic-clearance batch."
            "</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    map_column, result_column = st.columns(
        [1.0, 1.0],
        gap="large",
    )

    with map_column:
        figure = render_wafer_map(
            np.asarray(result["wafer_map"]),
            title=st.session_state.get(
                "upload_classifier_source_name",
                "Uploaded wafer",
            ),
        )
        st.pyplot(
            figure,
            clear_figure=True,
            use_container_width=True,
        )
        plt.close(figure)

    with result_column:
        st.markdown("#### Model probabilities")
        for rank, prediction in enumerate(
            result["top_predictions"],
            start=1,
        ):
            probability = float(
                prediction["probability"]
            )
            left, right = st.columns([3.3, 1.0])
            with left:
                st.caption(
                    f"{rank}. {prediction['class_name']}"
                )
                st.progress(
                    float(np.clip(probability, 0.0, 1.0))
                )
            with right:
                st.caption(f"{probability:.2%}")

        failed_dies = int(result["failed_dies"])
        active_dies = int(result["active_dies"])
        failed_ratio = (
            failed_dies / active_dies
            if active_dies
            else 0.0
        )
        metadata = [
            ("Source shape", str(result["raw_shape"])),
            ("Model tensor", str(result["model_shape"])),
            ("Active pixels", f"{active_dies:,}"),
            ("Failing pixels", f"{failed_dies:,}"),
            ("Failing ratio", f"{failed_ratio:.2%}"),
            ("Runtime device", str(run_device)),
        ]
        metadata_html = "".join(
            (
                f'<div class="label">{html.escape(label)}</div>'
                f'<div class="value">{html.escape(value)}</div>'
            )
            for label, value in metadata
        )
        st.markdown(
            f'<div class="upload-metadata">{metadata_html}</div>',
            unsafe_allow_html=True,
        )



def render_sidebar_quick_upload(
    *,
    base: ModuleType,
    manifest: dict[str, Any],
    model: torch.nn.Module,
    run_device: torch.device,
    step11: Any,
) -> None:
    """Render a compact upload-classification shortcut in the sidebar."""
    with st.sidebar:
        with st.expander(
            "Quick wafer upload",
            expanded=False,
        ):
            st.caption(
                "Classify one wafer map using the frozen CNN. "
                "Accepted: NPY, NPZ, or clean Kavach palette PNG."
            )

            uploaded = st.file_uploader(
                "Wafer-map file",
                type=["npy", "npz", "png"],
                accept_multiple_files=False,
                key="sidebar_quick_upload_file",
                help=(
                    "Raw maps must use 0=inactive, 1=passing, "
                    "2=failing."
                ),
            )

            payload: bytes | None = None
            npz_key: str | None = None

            if uploaded is not None:
                payload = uploaded.getvalue()

                if Path(uploaded.name).suffix.lower() == ".npz":
                    try:
                        with np.load(
                            BytesIO(payload),
                            allow_pickle=False,
                        ) as archive:
                            available_keys = list(archive.files)
                    except Exception as exc:
                        st.error(
                            "Unable to inspect NPZ: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        available_keys = []

                    if available_keys:
                        npz_key = st.selectbox(
                            "NPZ array",
                            available_keys,
                            key="sidebar_quick_upload_npz_key",
                        )

            acknowledgement = st.checkbox(
                "Treat this upload as an engineer-review case.",
                key="sidebar_quick_upload_ack",
            )

            run_clicked = st.button(
                "Classify now",
                type="primary",
                use_container_width=True,
                disabled=(
                    uploaded is None
                    or not acknowledgement
                ),
                key="sidebar_quick_upload_run",
            )

            if (
                run_clicked
                and uploaded is not None
                and payload is not None
            ):
                try:
                    raw_wafer = parse_uploaded_wafer(
                        uploaded_name=uploaded.name,
                        payload=payload,
                        npz_key=npz_key,
                    )
                    result = classify_uploaded_wafer(
                        base=base,
                        raw_wafer=raw_wafer,
                        model=model,
                        run_device=run_device,
                        step11=step11,
                        class_names=[
                            str(value)
                            for value in manifest[
                                "model"
                            ]["class_names"]
                        ],
                    )
                    st.session_state[
                        "sidebar_quick_upload_result"
                    ] = result
                    st.session_state[
                        "sidebar_quick_upload_name"
                    ] = uploaded.name
                except Exception as exc:
                    st.session_state.pop(
                        "sidebar_quick_upload_result",
                        None,
                    )
                    st.error(
                        "Classification failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            result = st.session_state.get(
                "sidebar_quick_upload_result"
            )

            if result is not None:
                predicted_class = str(
                    result["predicted_class"]
                )
                confidence = float(
                    result["confidence"]
                )
                margin = float(result["margin"])

                st.markdown(
                    (
                        "**Result**  \n"
                        f"Class: **{predicted_class}**  \n"
                        f"Confidence: **{confidence:.2%}**  \n"
                        f"Top-1/Top-2 margin: **{margin:.2%}**"
                    )
                )

                st.caption(
                    "Operational route: ENGINEER_REVIEW. "
                    "Uploaded files are never auto-cleared."
                )

                for rank, prediction in enumerate(
                    result["top_predictions"],
                    start=1,
                ):
                    probability = float(
                        prediction["probability"]
                    )
                    st.caption(
                        f"{rank}. "
                        f"{prediction['class_name']} — "
                        f"{probability:.2%}"
                    )

                st.caption(
                    "Open the Upload classifier tab for the "
                    "full wafer view and detailed evidence."
                )



def main() -> None:
    st.set_page_config(
        page_title="Kavach AI | Wafer Decisions",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cli = parse_cli_args()

    try:
        base = load_base_dashboard(cli.base_dashboard)
        base.inject_styles()
        inject_ledger_styles()

        bundle = base.resolve_bundle(cli.bundle)
        manifest_path = (
            bundle / "REVIEW_BUNDLE_MANIFEST.json"
        )
        manifest_hash = base.sha256_file(manifest_path)

        (
            manifest,
            base_queue,
            tensors,
            lookup,
            model,
            run_device,
            step11,
        ) = base.load_runtime(
            str(bundle),
            manifest_hash,
        )

        database_path = (
            cli.database.expanduser().resolve()
        )
        base.initialise_database(database_path)

        chain_ok, chain_message = base.verify_chain(
            database_path
        )
        require(
            chain_ok,
            f"Audit-chain verification failed: {chain_message}",
        )

        inference_path = (
            Path(str(manifest["source_inference_run"]))
            .expanduser()
            .resolve()
            / "inference_results.csv"
        )
        overlay_path = (
            Path(str(manifest["source_drift_run"]))
            .expanduser()
            .resolve()
            / "operational_routing_overlay.csv"
        )

        source_hashes = manifest["source_hashes"]
        ledger = load_governed_ledger(
            str(inference_path),
            str(source_hashes["inference_results_csv"]),
            str(overlay_path),
            str(source_hashes["routing_overlay"]),
            len(manifest["model"]["class_names"]),
        )

        latest = base.latest_events(
            database_path,
            str(manifest["bundle_id"]),
        )
        queue = base.queue_with_review_status(
            base_queue,
            latest,
        )
        ledger = attach_decision_status(
            ledger,
            latest,
        )
        validate_ledger_contract(
            ledger,
            queue,
            manifest,
        )

    except Exception as exc:
        st.error(
            "Decision Ledger preflight failed: "
            f"{type(exc).__name__}: {exc}"
        )
        st.stop()

    filters = base.render_sidebar(
        manifest=manifest,
        queue=queue,
        chain_message=chain_message,
    )

    render_sidebar_quick_upload(
        base=base,
        manifest=manifest,
        model=model,
        run_device=run_device,
        step11=step11,
    )

    filtered_queue = base.filter_queue(
        queue,
        **filters,
    )

    render_system_header(
        base,
        manifest,
        chain_message,
    )

    st.caption(
        "Public portfolio demonstration using historical WM-811K wafer-map "
        "data. This application is not connected to a live fabrication "
        "facility. Review events use local demo storage and may reset after "
        "a cloud restart."
    )

    flash_message = st.session_state.pop(
        "kavach_flash_success",
        None,
    )
    if flash_message:
        st.success(flash_message)

    total_decisions = len(ledger)
    auto_cleared = int(
        ledger["effective_route"].eq("AUTO_ACCEPT").sum()
    )
    review_required = int(
        ledger["effective_route"].isin(REVIEW_ROUTES).sum()
    )
    pending = int(
        queue["review_status"].eq("PENDING").sum()
    )
    reviewed = int(
        queue["review_status"].eq("REVIEWED").sum()
    )

    top_metrics = st.columns(5, gap="small")
    top_metrics[0].metric(
        "Inference batch",
        f"{total_decisions:,}",
    )
    top_metrics[1].metric(
        "Auto-cleared",
        f"{auto_cleared:,}",
    )
    top_metrics[2].metric(
        "Review queue",
        f"{review_required:,}",
    )
    top_metrics[3].metric(
        "Pending",
        f"{pending:,}",
    )
    top_metrics[4].metric(
        "Reviewed",
        f"{reviewed:,}",
    )

    (
        review_tab,
        ledger_tab,
        upload_tab,
        analytics_tab,
        audit_tab,
        governance_tab,
    ) = st.tabs(
        [
            "Review workbench",
            "Decision ledger",
            "Upload classifier",
            "Queue analytics",
            "Audit trail",
            "Governance",
        ]
    )

    with review_tab:
        base.render_review_workbench(
            filtered=filtered_queue,
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

    with ledger_tab:
        render_decision_ledger(
            ledger,
            tensors=tensors,
            lookup=lookup,
            step11=step11,
        )

    with upload_tab:
        render_upload_classifier(
            base=base,
            manifest=manifest,
            model=model,
            run_device=run_device,
            step11=step11,
        )

    with analytics_tab:
        base.render_queue_analytics(queue)

    with audit_tab:
        base.render_audit_trail(
            database_path=database_path,
            manifest=manifest,
        )

    with governance_tab:
        base.render_governance(
            manifest=manifest,
            bundle=bundle,
            database_path=database_path,
        )


if __name__ == "__main__":
    main()
