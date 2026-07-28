"""Streamlit interface for multimodal tabular + text supervised learning."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from data_validation import (
    DataQualityReport,
    combine_reports,
    inspect_excel_sheet,
    validate_dataframe,
)
from multimodal_pipeline import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    ExperimentConfig,
    infer_task,
    run_experiment,
)

st.set_page_config(
    page_title="Semantic Fusion Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGGER = logging.getLogger(__name__)
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_DATA_ROWS = 50_000
MAX_DATA_COLUMNS = 500
LARGE_DATA_WARNING_ROWS = 10_000

MODEL_OPTIONS = {
    "all-mpnet-base-v2 — strong English option (768D)": (
        "sentence-transformers/all-mpnet-base-v2"
    ),
    "all-MiniLM-L6-v2 — faster, smaller English option (384D)": (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    "paraphrase-multilingual-MiniLM-L12-v2 — multilingual (384D)": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "bge-small-en-v1.5 — compact English embedding model (384D)": (
        "BAAI/bge-small-en-v1.5"
    ),
    "Custom Hugging Face / Sentence Transformers model": "__custom__",
}


def _style_page() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #607089;
            --teal: #087f75;
            --teal-soft: #e6f5f2;
            --line: #dce4ec;
            --paper: #ffffff;
        }
        .stApp {
            background:
                radial-gradient(
                    circle at 88% 3%,
                    rgba(8,127,117,.10),
                    transparent 24rem
                ),
                linear-gradient(180deg, #f7fafc 0%, #ffffff 42%);
        }
        .hero {
            padding: 1.7rem 1.9rem;
            border: 1px solid var(--line);
            border-radius: 20px;
            background: rgba(255,255,255,.92);
            box-shadow: 0 14px 38px rgba(23,32,51,.07);
            margin-bottom: 1.2rem;
        }
        .hero-kicker {
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: .12em;
            font-weight: 750;
            font-size: .75rem;
        }
        .hero h1 {
            color: var(--ink);
            margin: .35rem 0 .55rem 0;
            font-size: clamp(2rem, 5vw, 3.4rem);
            line-height: 1.03;
        }
        .hero p {
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 58rem;
            margin: 0;
        }
        .step-label {
            color: var(--teal);
            font-size: .78rem;
            font-weight: 750;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-top: .5rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: .8rem 1rem;
            background: rgba(255,255,255,.90);
        }
        div[data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: rgba(255,255,255,.78);
        }
        .fine-print {
            color: var(--muted);
            font-size: .82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _excel_sheet_names(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


@st.cache_data(show_spinner=False)
def _read_excel(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


@st.cache_resource(show_spinner=False)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=False,
            local_files_only=True,
        )
    except (OSError, RuntimeError):
        # Fresh deployments do not yet have a cache, so allow the normal Hub
        # download only after the cache-only attempt fails.
        model = SentenceTransformer(model_name, trust_remote_code=False)
    return model, threading.RLock()


class _CachedEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model, self.lock = _load_sentence_transformer(model_name)

    def encode(self, sentences, **kwargs):
        # Streamlit shares cached resources across sessions. Serializing calls
        # prevents concurrent CPU inference from overcommitting memory.
        with self.lock:
            return self.model.encode(sentences, **kwargs)


def _dataset_from_interface() -> tuple[
    pd.DataFrame | None,
    str | None,
    str | None,
    DataQualityReport | None,
]:
    uploaded = st.file_uploader(
        "Upload all data as CSV or Excel",
        type=["csv", "xlsx", "xlsm"],
        help=(
            "Each row must be one observation and each column one variable. "
            "The app does not use server-side or bundled datasets."
        ),
    )
    if uploaded is None:
        return None, None, None, None
    file_bytes = uploaded.getvalue()
    file_name = uploaded.name

    if len(file_bytes) > MAX_FILE_BYTES:
        st.error(
            f"The file is larger than the {MAX_FILE_BYTES // (1024**2)} MB "
            "application limit."
        )
        return None, None, None, None

    try:
        if file_name.lower().endswith(".csv"):
            frame = _read_csv(file_bytes)
            source_label = file_name
            source_hash = hashlib.sha256(file_bytes).hexdigest()
            quality_report = validate_dataframe(frame)
        else:
            sheet_names = _excel_sheet_names(file_bytes)
            sheet_name = st.selectbox(
                "Worksheet",
                sheet_names,
                index=None,
                placeholder="Select a worksheet",
                help="Choose the workbook sheet that contains the analysis table.",
            )
            if sheet_name is None:
                st.info("Select a worksheet to inspect its data.")
                return None, None, None, None
            frame = _read_excel(file_bytes, sheet_name)
            source_label = f"{file_name} · {sheet_name}"
            source_hash = hashlib.sha256(
                file_bytes + sheet_name.encode("utf-8")
            ).hexdigest()
            quality_report = combine_reports(
                inspect_excel_sheet(file_bytes, sheet_name),
                validate_dataframe(frame),
            )
    except Exception as exc:
        st.error(f"Could not read the uploaded data: {exc}")
        return None, None, None, None

    if len(frame) > MAX_DATA_ROWS:
        st.error(
            f"This deployment accepts up to {MAX_DATA_ROWS:,} rows per run; "
            f"the selected data has {len(frame):,}. Sample or partition it first."
        )
        return None, None, None, None
    if len(frame.columns) > MAX_DATA_COLUMNS:
        st.error(
            f"This deployment accepts up to {MAX_DATA_COLUMNS:,} columns; "
            f"the selected data has {len(frame.columns):,}."
        )
        return None, None, None, None
    return frame, source_label, source_hash, quality_report


def _render_data_quality(report: DataQualityReport) -> None:
    if not report.issues:
        st.success("Data quality checks passed with no detected issues.")
        return

    st.markdown("#### Data quality checks")
    for issue in report.issues:
        details: list[str] = []
        if issue.affected_count:
            details.append(f"Affected: {issue.affected_count:,}")
        if issue.columns:
            details.append("Columns: " + ", ".join(issue.columns))
        if issue.examples:
            details.append("Examples: " + ", ".join(issue.examples))
        message = issue.message
        if details:
            message += " " + " · ".join(details)
        if issue.severity == "error":
            st.error(message)
        else:
            st.warning(message)


def _recommended_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    text_candidates: list[str] = []
    tabular_candidates: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            lengths = series.dropna().astype(str).str.len()
            if not lengths.empty and (lengths.median() >= 40 or lengths.max() >= 200):
                text_candidates.append(column)
                continue
        tabular_candidates.append(column)
    return text_candidates, tabular_candidates


def _model_parameter_controls(
    model_names: list[str],
) -> dict[str, dict[str, int | float]]:
    parameters: dict[str, dict[str, int | float]] = {}
    for model_name in model_names:
        with st.expander(f"{model_name} training parameters", expanded=True):
            left, right = st.columns(2)
            key_prefix = model_name.lower().replace(" ", "_")
            if model_name == "Logistic Regression":
                with left:
                    regularization = st.number_input(
                        "Inverse regularization strength (C)",
                        min_value=0.001,
                        max_value=1_000.0,
                        value=1.0,
                        step=0.1,
                        format="%.3f",
                        key=f"{key_prefix}_c",
                        help=(
                            "Smaller values apply stronger regularization; larger "
                            "values let the model fit the training data more closely."
                        ),
                    )
                with right:
                    max_iter = st.slider(
                        "Maximum training iterations",
                        100,
                        5_000,
                        3_000,
                        step=100,
                        key=f"{key_prefix}_max_iter",
                        help=(
                            "Maximum optimization steps allowed before training "
                            "stops. Increase it if convergence warnings appear."
                        ),
                    )
                parameters[model_name] = {
                    "C": float(regularization),
                    "max_iter": int(max_iter),
                }
            elif model_name == "Ridge Regression":
                alpha = st.number_input(
                    "Regularization strength (alpha)",
                    min_value=0.0,
                    max_value=1_000.0,
                    value=1.0,
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_alpha",
                    help=(
                        "Higher values shrink coefficients more strongly and can "
                        "reduce overfitting."
                    ),
                )
                parameters[model_name] = {"alpha": float(alpha)}
            elif model_name == "Random Forest":
                with left:
                    n_estimators = st.slider(
                        "Number of trees",
                        50,
                        1_000,
                        350,
                        step=50,
                        key=f"{key_prefix}_n_estimators",
                        help=(
                            "More trees usually stabilize predictions but increase "
                            "training time and memory use."
                        ),
                    )
                    min_samples_leaf = st.slider(
                        "Minimum samples per leaf",
                        1,
                        50,
                        1,
                        key=f"{key_prefix}_min_samples_leaf",
                        help=(
                            "Larger leaves produce smoother, more regularized trees."
                        ),
                    )
                with right:
                    max_depth = st.slider(
                        "Maximum tree depth",
                        0,
                        50,
                        0,
                        key=f"{key_prefix}_max_depth",
                        help=(
                            "Limits tree depth. A value of 0 allows trees to expand "
                            "until other stopping rules apply."
                        ),
                    )
                    max_features = st.slider(
                        "Feature fraction per split",
                        0.1,
                        1.0,
                        1.0,
                        step=0.1,
                        key=f"{key_prefix}_max_features",
                        help=(
                            "Fraction of available predictors considered at each "
                            "split. Lower values increase diversity among trees."
                        ),
                    )
                parameters[model_name] = {
                    "n_estimators": int(n_estimators),
                    "max_depth": int(max_depth),
                    "min_samples_leaf": int(min_samples_leaf),
                    "max_features": float(max_features),
                }
            elif model_name == "XGBoost":
                with left:
                    n_estimators = st.slider(
                        "Boosting rounds",
                        50,
                        1_000,
                        350,
                        step=50,
                        key=f"{key_prefix}_n_estimators",
                        help="Number of sequential boosted trees to train.",
                    )
                    max_depth = st.slider(
                        "Maximum tree depth",
                        1,
                        15,
                        6,
                        key=f"{key_prefix}_max_depth",
                        help=(
                            "Maximum depth of each boosted tree. Deeper trees can "
                            "model more complex interactions but may overfit."
                        ),
                    )
                    learning_rate = st.slider(
                        "Learning rate",
                        0.01,
                        0.50,
                        0.05,
                        step=0.01,
                        key=f"{key_prefix}_learning_rate",
                        help=(
                            "Contribution of each new tree. Smaller values usually "
                            "need more boosting rounds."
                        ),
                    )
                with right:
                    subsample = st.slider(
                        "Row sampling fraction",
                        0.50,
                        1.00,
                        0.85,
                        step=0.05,
                        key=f"{key_prefix}_subsample",
                        help="Fraction of training rows used for each boosted tree.",
                    )
                    colsample = st.slider(
                        "Feature sampling fraction",
                        0.50,
                        1.00,
                        0.85,
                        step=0.05,
                        key=f"{key_prefix}_colsample",
                        help=(
                            "Fraction of predictors considered when constructing "
                            "each boosted tree."
                        ),
                    )
                parameters[model_name] = {
                    "n_estimators": int(n_estimators),
                    "max_depth": int(max_depth),
                    "learning_rate": float(learning_rate),
                    "subsample": float(subsample),
                    "colsample_bytree": float(colsample),
                }
            elif model_name == "CatBoost":
                with left:
                    iterations = st.slider(
                        "Boosting iterations",
                        50,
                        1_000,
                        350,
                        step=50,
                        key=f"{key_prefix}_iterations",
                        help="Number of sequential boosting iterations to train.",
                    )
                    depth = st.slider(
                        "Tree depth",
                        2,
                        12,
                        6,
                        key=f"{key_prefix}_depth",
                        help=(
                            "Depth of each CatBoost tree. Greater depth captures "
                            "more complex patterns but uses more memory."
                        ),
                    )
                with right:
                    learning_rate = st.slider(
                        "Learning rate",
                        0.01,
                        0.50,
                        0.05,
                        step=0.01,
                        key=f"{key_prefix}_learning_rate",
                        help=(
                            "Contribution of each boosting iteration. Smaller "
                            "values normally require more iterations."
                        ),
                    )
                    l2_leaf_reg = st.slider(
                        "L2 leaf regularization",
                        0.0,
                        20.0,
                        3.0,
                        step=0.5,
                        key=f"{key_prefix}_l2_leaf_reg",
                        help=(
                            "Penalty applied to leaf values; higher values can "
                            "reduce overfitting."
                        ),
                    )
                parameters[model_name] = {
                    "iterations": int(iterations),
                    "depth": int(depth),
                    "learning_rate": float(learning_rate),
                    "l2_leaf_reg": float(l2_leaf_reg),
                }
    return parameters


def _csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _safe_csv(frame: pd.DataFrame, *, include_index: bool = False) -> str:
    safe = frame.copy()
    safe.columns = [_csv_safe(str(column)) for column in safe.columns]
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(_csv_safe)
    if include_index:
        safe.index = [_csv_safe(str(value)) for value in safe.index]
    return safe.to_csv(index=include_index)


def _build_download_bundle(result) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metrics.csv", _safe_csv(result.metrics))
        archive.writestr("predictions.csv", _safe_csv(result.predictions))
        archive.writestr(
            "configuration.json",
            json.dumps(result.configuration, indent=2, default=str),
        )
        archive.writestr(
            "diagnostics.json",
            json.dumps(result.diagnostics, indent=2, default=str),
        )
        for model_name, matrix in result.confusion_matrices.items():
            safe_name = (
                model_name.lower()
                .replace(" · ", "__")
                .replace(" ", "_")
            )
            archive.writestr(
                f"confusion_matrix_{safe_name}.csv",
                _safe_csv(matrix, include_index=True),
            )
        if result.embeddings is not None:
            embedding_buffer = io.BytesIO()
            np.savez_compressed(
                embedding_buffer,
                embeddings=result.embeddings,
                source_row_indices=np.asarray(
                    result.embedding_row_indices,
                    dtype=str,
                ),
            )
            archive.writestr(
                "sentence_embeddings.npz",
                embedding_buffer.getvalue(),
            )
    return buffer.getvalue()


def _render_results(result) -> None:
    st.markdown("---")
    st.subheader("Results")
    diagnostics = result.diagnostics
    columns = st.columns(5)
    columns[0].metric("Training rows", f"{diagnostics['training_rows']:,}")
    columns[1].metric("Test rows", f"{diagnostics['test_rows']:,}")
    columns[2].metric(
        "Embedding dimensions",
        f"{diagnostics['embedding_dimensions_used']:,}",
        delta=(
            diagnostics["embedding_dimensions_used"]
            - diagnostics["embedding_dimensions_original"]
        ),
        delta_color="off",
    )
    columns[3].metric(
        "Tabular dimensions",
        f"{diagnostics['tabular_dimensions_after_encoding']:,}",
    )
    columns[4].metric(
        "Total model features",
        f"{diagnostics['total_model_features']:,}",
    )
    st.caption(
        f"Split: {diagnostics['split_method']} · "
        f"PCA retained {diagnostics['pca_explained_variance']:.1%} "
        "of embedding variance · "
        f"{diagnostics['feature_matrix_format']} training matrix "
        f"({diagnostics['training_feature_memory_mb']:.1f} MB)."
    )
    for warning in diagnostics.get("warnings", []):
        st.warning(warning)

    if diagnostics.get("baseline_analysis_included"):
        st.info(
            "The tabular baseline uses the same analyzed rows, split, target, "
            "tabular predictors, models, hyperparameters, and evaluation metrics "
            "as the multimodal analysis, but excludes all text embeddings."
        )

    st.dataframe(
        result.metrics.style.format(precision=4),
        width="stretch",
        hide_index=True,
    )

    if result.confusion_matrices:
        st.markdown("#### Confusion matrices")
        tabs = st.tabs(list(result.confusion_matrices))
        for tab, (_model_name, matrix) in zip(
            tabs, result.confusion_matrices.items(), strict=True
        ):
            with tab:
                st.dataframe(matrix, width="stretch")

    with st.expander("Test-set predictions", expanded=False):
        st.dataframe(result.predictions, width="stretch", hide_index=True)

    st.download_button(
        "Download results bundle (.zip)",
        data=_build_download_bundle(result),
        file_name="semantic_fusion_results.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
        help=(
            "Download metrics, predictions, configuration, diagnostics, confusion "
            "matrices, and optional retained embeddings."
        ),
    )


def main() -> None:
    _style_page()
    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">Multimodal supervised learning</div>
          <h1>Semantic Fusion Lab</h1>
          <p>
            Combine tabular and text predictors using sentence-transformer
            embeddings, optionally compress the semantic space with PCA, and
            compare classification or regression models—without writing code.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Workflow")
        st.markdown(
            "1. Upload data\n"
            "2. Assign variable roles\n"
            "3. Choose an embedding model\n"
            "4. Configure PCA and prediction\n"
            "5. Run and download results"
        )
        st.info(
            "For repeated entities (patients, customers, papers), select a group "
            "column so the same entity does not appear in both train and test sets."
        )

    st.markdown('<div class="step-label">Step 1 · Data</div>', unsafe_allow_html=True)
    frame, source_label, source_hash, quality_report = _dataset_from_interface()
    if frame is None:
        st.info("Upload a complete dataset to begin.")
        return

    st.success(
        f"Loaded {source_label}: {len(frame):,} rows × {len(frame.columns):,} columns."
    )
    if len(frame) > LARGE_DATA_WARNING_ROWS:
        st.warning(
            "This is a large CPU workload. Start with the MiniLM model, a smaller "
            "sample, and one prediction model before scaling up."
        )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(frame):,}")
    metric_columns[1].metric("Columns", f"{len(frame.columns):,}")
    metric_columns[2].metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
    metric_columns[3].metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
    with st.expander("Preview and column profile", expanded=True):
        st.dataframe(frame.head(20), width="stretch", hide_index=True)
        profile = pd.DataFrame(
            {
                "column": frame.columns,
                "dtype": [str(frame[column].dtype) for column in frame.columns],
                "missing": [
                    int(frame[column].isna().sum()) for column in frame.columns
                ],
                "unique": [
                    int(frame[column].nunique(dropna=True)) for column in frame.columns
                ],
            }
        )
        st.dataframe(profile, width="stretch", hide_index=True)

    if quality_report is None:
        st.error("Data quality checks could not be completed.")
        return
    _render_data_quality(quality_report)
    if not quality_report.can_proceed:
        st.error(
            "Analysis is blocked. Correct the errors above and upload the "
            "dataset again."
        )
        return
    if quality_report.warnings:
        warnings_reviewed = st.checkbox(
            "I reviewed the data-quality warnings and confirm the flagged "
            "values are appropriate for this analysis.",
            value=False,
            help=(
                "Acknowledge that you inspected the reported zeros, missing "
                "values, duplicate rows, or constant columns before continuing."
            ),
        )
        if not warnings_reviewed:
            st.info("Review and acknowledge the warnings to configure the analysis.")
            return

    suggested_text, _ = _recommended_columns(frame)
    all_columns = list(frame.columns)

    st.markdown(
        '<div class="step-label">Steps 2–4 · Configure analysis</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("#### Variable roles")
        role_left, role_right = st.columns(2)
        with role_left:
            target_column = st.selectbox(
                "Dependent variable (target)",
                all_columns,
                index=None,
                placeholder="Select a target column",
                help=(
                    "The outcome the models will learn to classify or predict."
                ),
            )
            if target_column is not None:
                suggested_task = infer_task(frame[target_column])
                st.caption(f"Suggested from target values: {suggested_task}.")
            task = st.selectbox(
                "Prediction task",
                ["classification", "regression"],
                index=None,
                placeholder="Select a prediction task",
                help=(
                    "Choose classification for discrete labels or regression for "
                    "a continuous numeric outcome."
                ),
            )
            group_options = ["Do not group rows"] + [
                column for column in all_columns if column != target_column
            ]
            group_choice = st.selectbox(
                "Group column for leakage-safe splitting",
                group_options,
                index=None,
                placeholder="Select a grouping option",
                help=(
                    "Rows sharing a group stay together in either training "
                    "or test data."
                ),
            )
            group_column = (
                None
                if group_choice in (None, "Do not group rows")
                else group_choice
            )

        eligible_predictors = [
            column
            for column in all_columns
            if column != target_column and column != group_column
        ]
        with role_right:
            text_columns = st.multiselect(
                "Natural-language text columns",
                eligible_predictors,
                default=[],
                placeholder="Select one or more text columns",
                help=(
                    "These columns are joined row by row and converted into "
                    "sentence embeddings."
                ),
            )
            suggested_available = [
                column for column in suggested_text if column in eligible_predictors
            ]
            if suggested_available:
                st.caption(
                    "Likely text columns: " + ", ".join(suggested_available[:5])
                )
            available_tabular = [
                column for column in eligible_predictors if column not in text_columns
            ]
            tabular_columns = st.multiselect(
                "Structured tabular predictors",
                available_tabular,
                default=[],
                placeholder="Select one or more tabular columns",
                help=(
                    "Numeric or categorical variables that will be preprocessed "
                    "and combined with the text embeddings."
                ),
            )

        st.markdown("#### Sentence transformer")
        embed_left, embed_right = st.columns([2, 1])
        with embed_left:
            model_labels = list(MODEL_OPTIONS)
            model_label = st.selectbox(
                "Embedding model",
                model_labels,
                index=None,
                placeholder="Select an embedding model",
                help=(
                    "Choose the pretrained sentence transformer used to convert "
                    "the selected text into numeric embedding vectors."
                ),
            )
            transformer_model = ""
            if (
                model_label is not None
                and MODEL_OPTIONS[model_label] == "__custom__"
            ):
                transformer_model = st.text_input(
                    "Model identifier",
                    placeholder="organization/model-name",
                    help="Only use models you trust. Remote custom code is disabled.",
                ).strip()
            elif model_label is not None:
                transformer_model = MODEL_OPTIONS[model_label]
        with embed_right:
            batch_size = st.select_slider(
                "Embedding batch size",
                options=[4, 8, 16, 32, 64, 128],
                value=32,
                help=(
                    "The number of text rows encoded together in one pass. "
                    "Larger batches can be faster but use more memory; reduce "
                    "the value if the app runs out of memory."
                ),
            )
            normalize_embeddings = st.checkbox(
                "L2-normalize embeddings",
                value=False,
                help=(
                    "Scale every embedding vector to unit length before PCA and "
                    "model training."
                ),
            )
            retain_embeddings = st.checkbox(
                "Retain embeddings for download",
                value=False,
                help=(
                    "Disabled by default to reduce per-session memory. If enabled, "
                    "embeddings are added to the ZIP as a compressed NumPy file."
                ),
            )

        st.markdown("#### PCA")
        pca_columns = st.columns([1, 2])
        with pca_columns[0]:
            pca_label = st.selectbox(
                "PCA",
                ["No PCA", "Fixed components", "Explained variance"],
                index=None,
                placeholder="Select a PCA option",
                help=(
                    "Optionally reduce only the embedding dimensions. PCA is "
                    "fitted on training data and then applied to test data."
                ),
            )
        with pca_columns[1]:
            pca_mode: str | None = None
            pca_components: int | float | None = None
            if pca_label == "Fixed components":
                pca_mode = "fixed"
                pca_components = st.number_input(
                    "Components",
                    min_value=1,
                    max_value=768,
                    value=100,
                    step=1,
                    help=(
                        "Exact number of principal components retained from the "
                        "text embedding vectors."
                    ),
                )
            elif pca_label == "Explained variance":
                pca_mode = "variance"
                pca_components = (
                    st.slider(
                        "Variance retained",
                        min_value=50,
                        max_value=99,
                        value=95,
                        help=(
                            "Minimum percentage of training-set embedding variance "
                            "that the retained components should explain."
                        ),
                    )
                    / 100
                )
            else:
                if pca_label == "No PCA":
                    pca_mode = "none"
        st.markdown("#### Prediction model selection")
        if task == "classification":
            available_models = CLASSIFICATION_MODELS
        elif task == "regression":
            available_models = REGRESSION_MODELS
        else:
            available_models = ()
        model_names = st.multiselect(
            "Prediction models",
            available_models,
            default=[],
            placeholder="Select one or more models",
            help=(
                "Select one or more estimators to train on the same split and "
                "compare using held-out test metrics."
            ),
        )
        model_parameters = _model_parameter_controls(model_names)

        st.markdown("##### Training and evaluation settings")
        evaluation_columns = st.columns(3)
        with evaluation_columns[0]:
            test_size = (
                st.slider(
                    "Test-set size",
                    10,
                    40,
                    20,
                    step=5,
                    help=(
                        "Percentage of rows held out from training and used only "
                        "for final model evaluation."
                    ),
                )
                / 100
            )
        with evaluation_columns[1]:
            random_state = st.number_input(
                "Random seed",
                min_value=0,
                max_value=1_000_000,
                value=42,
                step=1,
                help=(
                    "Controls repeatable splitting and randomized model training."
                ),
            )
        with evaluation_columns[2]:
            stratify = st.checkbox(
                "Stratify classification split",
                value=True,
                disabled=task != "classification" or group_column is not None,
                help="Group-aware splitting takes precedence over stratification.",
            )
            st.caption(
                "PCA and tabular preprocessing are fitted on training data only."
            )

        st.markdown("##### Baseline comparison")
        run_tabular_baseline = st.checkbox(
            "Also run a tabular-only baseline analysis",
            value=False,
            help=(
                "Train each selected prediction model again using the same target, "
                "tabular predictors, rows, split, random seed, hyperparameters, and "
                "metrics, but exclude the text columns and sentence-transformer "
                "embeddings. This provides a direct comparison of whether semantic "
                "text features improve predictive performance."
            ),
        )

        submitted = st.button(
            "Run multimodal analysis",
            type="primary",
            width="stretch",
            help=(
                "Validate the selected configuration, generate embeddings, train "
                "the models, and calculate held-out results."
            ),
        )

    missing_selections: list[str] = []
    if target_column is None:
        missing_selections.append("dependent variable")
    if task is None:
        missing_selections.append("prediction task")
    if group_choice is None:
        missing_selections.append("grouping option")
    if not text_columns:
        missing_selections.append("at least one text column")
    if not tabular_columns:
        missing_selections.append("at least one tabular predictor")
    if model_label is None:
        missing_selections.append("embedding model")
    elif not transformer_model:
        missing_selections.append("custom embedding model identifier")
    if pca_label is None:
        missing_selections.append("PCA option")
    if not model_names:
        missing_selections.append("at least one prediction model")

    configuration: ExperimentConfig | None = None
    analysis_signature: str | None = None
    if not missing_selections:
        configuration = ExperimentConfig(
            target_column=target_column,
            task=task,
            text_columns=tuple(text_columns),
            tabular_columns=tuple(tabular_columns),
            model_names=tuple(model_names),
            transformer_model=transformer_model,
            group_column=group_column,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=bool(stratify),
            normalize_embeddings=bool(normalize_embeddings),
            embedding_batch_size=int(batch_size),
            pca_mode=pca_mode,
            pca_components=pca_components,
            retain_embeddings=bool(retain_embeddings),
            run_tabular_baseline=bool(run_tabular_baseline),
            model_parameters=model_parameters,
        )
        analysis_signature = hashlib.sha256(
            json.dumps(
                {
                    "source_hash": source_hash,
                    "configuration": configuration.__dict__,
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    if submitted:
        if missing_selections:
            st.error(
                "Select every required option before running the analysis: "
                + ", ".join(missing_selections)
                + "."
            )
        elif configuration is not None:
            try:
                with st.status(
                    "Running the multimodal pipeline…",
                    expanded=True,
                ) as status:
                    st.write(f"Loading `{transformer_model}`")
                    embedder = _CachedEmbedder(transformer_model)
                    st.write("Encoding text and preparing leakage-safe features")
                    result = run_experiment(frame, configuration, embedder)
                    st.write("Training selected models and calculating test metrics")
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_signature"] = analysis_signature
                    status.update(
                        label="Analysis complete",
                        state="complete",
                        expanded=False,
                    )
            except Exception as exc:
                LOGGER.exception("Multimodal analysis failed")
                st.error(f"Analysis could not be completed: {exc}")

    result = st.session_state.get("analysis_result")
    result_signature = st.session_state.get("analysis_signature")
    if (
        result is not None
        and analysis_signature is not None
        and result_signature == analysis_signature
    ):
        _render_results(result)
    elif result is not None:
        st.info(
            "The configuration has changed since the last run. Complete the "
            "selections and run the analysis again to generate matching results."
        )

    st.markdown(
        """
        <p class="fine-print">
        This application supports exploratory supervised-learning experiments.
        Performance estimates depend on data quality, split design, sample size,
        model selection, and domain validity; they are not a substitute for
        external validation.
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
