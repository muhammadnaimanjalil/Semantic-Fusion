"""Reusable multimodal supervised-learning pipeline.

The module intentionally contains no Streamlit code so the analytical workflow
can be tested, reused from notebooks, or exposed through another interface.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

TaskType = Literal["classification", "regression"]
PcaMode = Literal["none", "fixed", "variance"]

MAX_CATEGORIES_PER_COLUMN = 256
MAX_CONFUSION_MATRIX_CLASSES = 50
MAX_PROBABILITY_CLASSES = 20
DEFAULT_N_JOBS = max(1, min(4, os.cpu_count() or 1))


class TextEmbedder(Protocol):
    """Minimal interface implemented by SentenceTransformer and test doubles."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ExperimentConfig:
    target_column: str
    task: TaskType
    text_columns: tuple[str, ...]
    tabular_columns: tuple[str, ...]
    model_names: tuple[str, ...]
    transformer_model: str
    group_column: str | None = None
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    normalize_embeddings: bool = False
    embedding_batch_size: int = 32
    pca_mode: PcaMode = "none"
    pca_components: int | float | None = None
    retain_embeddings: bool = False
    model_parameters: dict[str, dict[str, int | float]] = field(
        default_factory=dict
    )


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    confusion_matrices: dict[str, pd.DataFrame]
    configuration: dict[str, Any]
    diagnostics: dict[str, Any]
    embeddings: np.ndarray | None
    embedding_row_indices: np.ndarray | None


def infer_task(series: pd.Series) -> TaskType:
    """Suggest classification for labels/low-cardinality values, else regression."""
    non_missing = series.dropna()
    if non_missing.empty:
        return "classification"
    if not pd.api.types.is_numeric_dtype(non_missing):
        return "classification"
    unique_count = non_missing.nunique()
    threshold = max(10, int(np.sqrt(len(non_missing))))
    return "classification" if unique_count <= threshold else "regression"


def combine_text_columns(frame: pd.DataFrame, text_columns: Sequence[str]) -> list[str]:
    """Combine one or more text columns while preserving column identity."""
    if len(text_columns) == 1:
        return frame[text_columns[0]].fillna("").astype(str).tolist()

    combined: list[str] = []
    for _, row in frame[list(text_columns)].iterrows():
        parts = [
            f"[{column}] {'' if pd.isna(row[column]) else str(row[column])}"
            for column in text_columns
        ]
        combined.append("\n".join(parts))
    return combined


def validate_configuration(frame: pd.DataFrame, config: ExperimentConfig) -> None:
    available = set(frame.columns)
    requested = {
        config.target_column,
        *config.text_columns,
        *config.tabular_columns,
    }
    if config.group_column:
        requested.add(config.group_column)
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Columns not found in the dataset: {', '.join(missing)}")
    if not config.text_columns:
        raise ValueError("Select at least one natural-language text column.")
    if not config.model_names:
        raise ValueError("Select at least one prediction model.")
    if (
        config.target_column in config.text_columns
        or config.target_column in config.tabular_columns
    ):
        raise ValueError(
            "The dependent variable cannot also be an independent variable."
        )
    overlap = set(config.text_columns) & set(config.tabular_columns)
    if overlap:
        raise ValueError(
            "Text and tabular roles must not overlap: " + ", ".join(sorted(overlap))
        )
    if config.group_column and (
        config.group_column in config.text_columns
        or config.group_column in config.tabular_columns
    ):
        raise ValueError(
            "The grouping column is used only for splitting and must not be "
            "a predictor."
        )
    if not 0.05 <= config.test_size <= 0.50:
        raise ValueError("Test size must be between 0.05 and 0.50.")
    if config.pca_mode == "fixed" and (
        not isinstance(config.pca_components, int) or config.pca_components < 1
    ):
        raise ValueError("Fixed PCA requires a positive integer component count.")
    if config.pca_mode == "variance":
        value = config.pca_components
        if not isinstance(value, float) or not 0.50 <= value < 1.0:
            raise ValueError(
                "Variance PCA requires a threshold from 0.50 to below 1.0."
            )


def _clean_target(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.Series, int]:
    if config.task == "regression":
        target = pd.to_numeric(frame[config.target_column], errors="coerce")
        valid = target.notna()
        cleaned_target = target.loc[valid].astype(float)
    else:
        valid = frame[config.target_column].notna()
        cleaned_target = frame.loc[valid, config.target_column].astype(str)
    dropped = int((~valid).sum())
    return frame.loc[valid].copy(), cleaned_target, dropped


def _can_stratify(target: pd.Series, test_size: float) -> bool:
    counts = target.value_counts()
    class_count = len(counts)
    test_count = int(np.ceil(len(target) * test_size))
    train_count = len(target) - test_count
    return bool(
        class_count > 1
        and counts.min() >= 2
        and test_count >= class_count
        and train_count >= class_count
    )


def _split_indices(
    frame: pd.DataFrame, target: pd.Series, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    positions = np.arange(len(frame))
    if config.group_column:
        groups = frame[config.group_column].copy()
        missing_mask = groups.isna()
        if missing_mask.any():
            replacement = [f"__missing_group_{i}" for i in positions[missing_mask]]
            groups.loc[missing_mask] = replacement
        # GroupShuffleSplit does not stratify. Try deterministic alternative
        # seeds so every class is represented in training when possible.
        all_classes = set(target) if config.task == "classification" else set()
        for offset in range(100):
            split_seed = config.random_state + offset
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=config.test_size,
                random_state=split_seed,
            )
            train_pos, test_pos = next(splitter.split(positions, target, groups))
            if (
                config.task != "classification"
                or set(target.iloc[train_pos]) == all_classes
            ):
                return (
                    train_pos,
                    test_pos,
                    f"group-aware (split seed {split_seed})",
                )
        raise ValueError(
            "A group-aware split could not place every target class in training. "
            "Consider collecting more groups for rare classes or changing test size."
        )

    stratification = None
    split_label = "random"
    if config.task == "classification" and config.stratify:
        if _can_stratify(target, config.test_size):
            stratification = target
            split_label = "stratified random"
        else:
            split_label = "random (stratification unavailable)"
    train_pos, test_pos = train_test_split(
        positions,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratification,
    )
    return np.asarray(train_pos), np.asarray(test_pos), split_label


def _build_tabular_preprocessor(
    frame: pd.DataFrame, columns: Sequence[str]
) -> ColumnTransformer | None:
    if not columns:
        return None

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]
    categorical_columns = [
        column for column in columns if column not in numeric_columns
    ]
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__missing__",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=2,
                        max_categories=MAX_CATEGORIES_PER_COLUMN,
                        sparse_output=True,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def _prepare_tabular_frame(
    frame: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    prepared = frame[list(columns)].copy()
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[column]):
            prepared[column] = prepared[column].dt.strftime("%Y-%m-%d")
    return prepared


def _combine_feature_blocks(tabular, embeddings: np.ndarray):
    if tabular.shape[1] == 0:
        return embeddings
    if sparse.issparse(tabular):
        return sparse.hstack(
            [tabular.tocsr(), sparse.csr_matrix(embeddings)],
            format="csr",
            dtype=np.float32,
        )
    return np.hstack([np.asarray(tabular, dtype=np.float32), embeddings])


def _is_finite(matrix) -> bool:
    values = matrix.data if sparse.issparse(matrix) else matrix
    return bool(np.isfinite(values).all())


def _matrix_nbytes(matrix) -> int:
    if sparse.issparse(matrix):
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return int(matrix.nbytes)


def _software_versions() -> dict[str, str]:
    packages = (
        "pandas",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "xgboost",
        "catboost",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _dataset_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    hashes = pd.util.hash_pandas_object(frame[list(columns)], index=True)
    return hashlib.sha256(hashes.to_numpy().tobytes()).hexdigest()


def _apply_pca(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, PCA | None]:
    if config.pca_mode == "none":
        return train_embeddings, test_embeddings, None

    max_components = min(
        train_embeddings.shape[0] - 1,
        train_embeddings.shape[1],
    )
    if max_components < 1:
        raise ValueError("There are too few training rows to fit PCA.")

    if config.pca_mode == "fixed":
        requested = int(config.pca_components)
        component_count: int | float = min(requested, max_components)
        solver = "auto"
    else:
        component_count = float(config.pca_components)
        solver = "full"

    reducer = PCA(
        n_components=component_count,
        svd_solver=solver,
        random_state=config.random_state,
    )
    reduced_train = reducer.fit_transform(train_embeddings)
    reduced_test = reducer.transform(test_embeddings)
    return reduced_train, reduced_test, reducer


def _build_estimator(
    model_name: str,
    task: TaskType,
    random_state: int,
    parameters: dict[str, int | float] | None = None,
):
    parameters = parameters or {}
    if task == "classification":
        if model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                C=float(parameters.get("C", 1.0)),
                max_iter=int(parameters.get("max_iter", 3_000)),
                class_weight="balanced",
                random_state=random_state,
            )
        if model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=int(parameters.get("n_estimators", 350)),
                max_depth=(
                    int(parameters["max_depth"])
                    if parameters.get("max_depth", 0)
                    else None
                ),
                min_samples_leaf=int(parameters.get("min_samples_leaf", 1)),
                max_features=float(parameters.get("max_features", 1.0)),
                class_weight="balanced",
                n_jobs=DEFAULT_N_JOBS,
                random_state=random_state,
            )
        if model_name == "XGBoost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "XGBoost is selected but the xgboost package is not installed."
                ) from exc
            return XGBClassifier(
                n_estimators=int(parameters.get("n_estimators", 350)),
                max_depth=int(parameters.get("max_depth", 6)),
                learning_rate=float(parameters.get("learning_rate", 0.05)),
                subsample=float(parameters.get("subsample", 0.85)),
                colsample_bytree=float(
                    parameters.get("colsample_bytree", 0.85)
                ),
                eval_metric="mlogloss",
                n_jobs=DEFAULT_N_JOBS,
                random_state=random_state,
            )
        if model_name == "CatBoost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:
                raise ImportError(
                    "CatBoost is selected but the catboost package is not installed."
                ) from exc
            return CatBoostClassifier(
                iterations=int(parameters.get("iterations", 350)),
                depth=int(parameters.get("depth", 6)),
                learning_rate=float(parameters.get("learning_rate", 0.05)),
                l2_leaf_reg=float(parameters.get("l2_leaf_reg", 3.0)),
                verbose=False,
                random_seed=random_state,
                allow_writing_files=False,
            )
    else:
        if model_name == "Ridge Regression":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=float(parameters.get("alpha", 1.0)))
        if model_name == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=int(parameters.get("n_estimators", 350)),
                max_depth=(
                    int(parameters["max_depth"])
                    if parameters.get("max_depth", 0)
                    else None
                ),
                min_samples_leaf=int(parameters.get("min_samples_leaf", 1)),
                max_features=float(parameters.get("max_features", 1.0)),
                n_jobs=DEFAULT_N_JOBS,
                random_state=random_state,
            )
        if model_name == "XGBoost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError(
                    "XGBoost is selected but the xgboost package is not installed."
                ) from exc
            return XGBRegressor(
                n_estimators=int(parameters.get("n_estimators", 350)),
                max_depth=int(parameters.get("max_depth", 6)),
                learning_rate=float(parameters.get("learning_rate", 0.05)),
                subsample=float(parameters.get("subsample", 0.85)),
                colsample_bytree=float(
                    parameters.get("colsample_bytree", 0.85)
                ),
                objective="reg:squarederror",
                n_jobs=DEFAULT_N_JOBS,
                random_state=random_state,
            )
        if model_name == "CatBoost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:
                raise ImportError(
                    "CatBoost is selected but the catboost package is not installed."
                ) from exc
            return CatBoostRegressor(
                iterations=int(parameters.get("iterations", 350)),
                depth=int(parameters.get("depth", 6)),
                learning_rate=float(parameters.get("learning_rate", 0.05)),
                l2_leaf_reg=float(parameters.get("l2_leaf_reg", 3.0)),
                loss_function="RMSE",
                verbose=False,
                random_seed=random_state,
                allow_writing_files=False,
            )
    raise ValueError(f"Unsupported {task} model: {model_name}")


def run_experiment(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    embedder: TextEmbedder,
) -> ExperimentResult:
    """Run one split and compare all selected prediction models."""
    validate_configuration(frame, config)
    clean_frame, target, dropped_target_rows = _clean_target(frame, config)
    if len(clean_frame) < 10:
        raise ValueError("At least 10 rows with a non-missing target are required.")
    if config.task == "classification" and target.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    original_indices = clean_frame.index.to_numpy()
    train_pos, test_pos, split_method = _split_indices(clean_frame, target, config)
    y_train_raw = target.iloc[train_pos]
    y_test_raw = target.iloc[test_pos]

    texts = combine_text_columns(clean_frame, config.text_columns)
    embeddings = np.asarray(
        embedder.encode(
            texts,
            batch_size=config.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=config.normalize_embeddings,
        ),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(clean_frame):
        raise ValueError(
            "The sentence transformer returned an unexpected embedding shape."
        )

    train_embeddings = embeddings[train_pos]
    test_embeddings = embeddings[test_pos]
    train_embeddings, test_embeddings, reducer = _apply_pca(
        train_embeddings,
        test_embeddings,
        config,
    )

    tabular_frame = _prepare_tabular_frame(clean_frame, config.tabular_columns)
    preprocessor = _build_tabular_preprocessor(tabular_frame, config.tabular_columns)
    if preprocessor is None:
        train_tabular = np.empty((len(train_pos), 0), dtype=np.float32)
        test_tabular = np.empty((len(test_pos), 0), dtype=np.float32)
    else:
        train_tabular = preprocessor.fit_transform(
            tabular_frame.iloc[train_pos]
        )
        test_tabular = preprocessor.transform(
            tabular_frame.iloc[test_pos]
        )

    x_train = _combine_feature_blocks(train_tabular, train_embeddings)
    x_test = _combine_feature_blocks(test_tabular, test_embeddings)
    if not _is_finite(x_train) or not _is_finite(x_test):
        raise ValueError("Prepared features contain non-finite values.")

    label_encoder: LabelEncoder | None = None
    if config.task == "classification":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        unseen_test = sorted(set(y_test_raw) - set(label_encoder.classes_))
        if unseen_test:
            raise ValueError(
                "The test split contains classes absent from training: "
                + ", ".join(map(str, unseen_test))
                + ". Try a different seed, stratification, or split."
            )
        y_test = label_encoder.transform(y_test_raw)
    else:
        y_train = y_train_raw.to_numpy(dtype=float)
        y_test = y_test_raw.to_numpy(dtype=float)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    confusion_frames: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    for model_name in config.model_names:
        estimator = _build_estimator(
            model_name,
            config.task,
            config.random_state,
            config.model_parameters.get(model_name),
        )
        estimator.fit(x_train, y_train)
        predicted = np.asarray(estimator.predict(x_test)).reshape(-1)

        if config.task == "classification":
            metric_rows.append(
                {
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, predicted),
                    "f1_weighted": f1_score(
                        y_test, predicted, average="weighted", zero_division=0
                    ),
                    "f1_macro": f1_score(
                        y_test, predicted, average="macro", zero_division=0
                    ),
                    "precision_weighted": precision_score(
                        y_test, predicted, average="weighted", zero_division=0
                    ),
                    "recall_weighted": recall_score(
                        y_test, predicted, average="weighted", zero_division=0
                    ),
                }
            )
            actual_labels = label_encoder.inverse_transform(y_test)
            predicted_labels = label_encoder.inverse_transform(
                np.asarray(predicted, dtype=int)
            )
            if len(label_encoder.classes_) <= MAX_CONFUSION_MATRIX_CLASSES:
                matrix = confusion_matrix(
                    actual_labels,
                    predicted_labels,
                    labels=label_encoder.classes_,
                )
                confusion_frames[model_name] = pd.DataFrame(
                    matrix,
                    index=[f"actual: {label}" for label in label_encoder.classes_],
                    columns=[f"predicted: {label}" for label in label_encoder.classes_],
                )
            elif not warnings:
                warnings.append(
                    "Confusion matrices were omitted because the target has more "
                    f"than {MAX_CONFUSION_MATRIX_CLASSES} classes."
                )
            prediction_frame = pd.DataFrame(
                {
                    "source_row_index": original_indices[test_pos],
                    "model": model_name,
                    "actual": actual_labels,
                    "predicted": predicted_labels,
                }
            )
            if (
                hasattr(estimator, "predict_proba")
                and len(label_encoder.classes_) <= MAX_PROBABILITY_CLASSES
            ):
                probabilities = estimator.predict_proba(x_test)
                for class_index, class_label in enumerate(label_encoder.classes_):
                    prediction_frame[f"probability_{class_label}"] = probabilities[
                        :, class_index
                    ]
            elif hasattr(estimator, "predict_proba") and not any(
                "Probability columns" in item for item in warnings
            ):
                warnings.append(
                    "Probability columns were omitted because the target has more "
                    f"than {MAX_PROBABILITY_CLASSES} classes."
                )
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
            metric_rows.append(
                {
                    "model": model_name,
                    "r2": r2_score(y_test, predicted),
                    "rmse": rmse,
                    "mae": mean_absolute_error(y_test, predicted),
                }
            )
            prediction_frame = pd.DataFrame(
                {
                    "source_row_index": original_indices[test_pos],
                    "model": model_name,
                    "actual": y_test,
                    "predicted": predicted,
                    "residual": y_test - predicted,
                }
            )
        prediction_frames.append(prediction_frame)

    pca_components = (
        int(reducer.n_components_) if reducer is not None else embeddings.shape[1]
    )
    explained_variance = (
        float(reducer.explained_variance_ratio_.sum()) if reducer is not None else 1.0
    )
    fingerprint_columns = list(
        dict.fromkeys(
            [
                config.target_column,
                *config.text_columns,
                *config.tabular_columns,
                *([config.group_column] if config.group_column else []),
            ]
        )
    )
    diagnostics = {
        "input_rows": int(len(frame)),
        "analyzed_rows": int(len(clean_frame)),
        "dropped_missing_target_rows": dropped_target_rows,
        "training_rows": int(len(train_pos)),
        "test_rows": int(len(test_pos)),
        "split_method": split_method,
        "embedding_dimensions_original": int(embeddings.shape[1]),
        "embedding_dimensions_used": pca_components,
        "pca_explained_variance": explained_variance,
        "tabular_dimensions_after_encoding": int(train_tabular.shape[1]),
        "total_model_features": int(x_train.shape[1]),
        "feature_matrix_format": "sparse CSR" if sparse.issparse(x_train) else "dense",
        "training_feature_memory_mb": _matrix_nbytes(x_train) / (1024**2),
        "dataset_fingerprint_sha256": _dataset_fingerprint(
            clean_frame, fingerprint_columns
        ),
        "software_versions": _software_versions(),
        "warnings": warnings,
        "train_source_indices": original_indices[train_pos].tolist(),
        "test_source_indices": original_indices[test_pos].tolist(),
    }
    return ExperimentResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        confusion_matrices=confusion_frames,
        configuration=asdict(config),
        diagnostics=diagnostics,
        embeddings=embeddings if config.retain_embeddings else None,
        embedding_row_indices=original_indices if config.retain_embeddings else None,
    )


CLASSIFICATION_MODELS = (
    "Logistic Regression",
    "XGBoost",
    "CatBoost",
    "Random Forest",
)

REGRESSION_MODELS = (
    "Ridge Regression",
    "XGBoost",
    "CatBoost",
    "Random Forest",
)
