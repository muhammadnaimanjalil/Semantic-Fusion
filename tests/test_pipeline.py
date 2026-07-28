from __future__ import annotations

import numpy as np
import pandas as pd

from multimodal_pipeline import (
    ExperimentConfig,
    _build_estimator,
    infer_task,
    run_experiment,
)


class FakeEmbedder:
    def encode(
        self,
        sentences,
        *,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
        normalize_embeddings,
    ):
        vectors = []
        for text in sentences:
            positive = text.lower().count("good")
            negative = text.lower().count("bad")
            vectors.append(
                [
                    len(text),
                    positive,
                    negative,
                    positive - negative,
                    sum(map(ord, text)) % 17,
                    text.count(" "),
                ]
            )
        output = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            output = output / np.where(norms == 0, 1, norms)
        return output


def _classification_frame() -> pd.DataFrame:
    rows = []
    for group in range(30):
        label = "accept" if group % 2 == 0 else "reject"
        for review in range(2):
            rows.append(
                {
                    "paper_id": group,
                    "review_id": review + 1,
                    "confidence": 3 + (group % 3),
                    "venue": "A" if group % 3 else "B",
                    "text": (
                        "good clear contribution"
                        if label == "accept"
                        else "bad unclear contribution"
                    ),
                    "decision": label,
                }
            )
    return pd.DataFrame(rows)


def test_infer_task():
    assert infer_task(pd.Series(["a", "b", "a"])) == "classification"
    assert infer_task(pd.Series(range(100))) == "regression"


def test_classification_group_split_has_no_group_overlap():
    frame = _classification_frame()
    config = ExperimentConfig(
        target_column="decision",
        task="classification",
        text_columns=("text",),
        tabular_columns=("review_id", "confidence", "venue"),
        model_names=("Logistic Regression",),
        transformer_model="fake",
        group_column="paper_id",
        pca_mode="fixed",
        pca_components=3,
        run_tabular_baseline=True,
    )
    result = run_experiment(frame, config, FakeEmbedder())
    train_groups = set(
        frame.loc[result.diagnostics["train_source_indices"], "paper_id"]
    )
    test_groups = set(frame.loc[result.diagnostics["test_source_indices"], "paper_id"])

    assert not train_groups & test_groups
    assert result.diagnostics["embedding_dimensions_original"] == 6
    assert result.diagnostics["embedding_dimensions_used"] == 3
    assert result.metrics.loc[0, "accuracy"] >= 0.5
    assert set(result.metrics["analysis"]) == {
        "Multimodal",
        "Tabular baseline",
    }
    assert len(result.predictions) == 2 * result.diagnostics["test_rows"]
    assert result.diagnostics["baseline_analysis_included"] is True
    assert result.diagnostics["baseline_model_features"] == 4
    assert set(result.predictions["actual"]) == {"accept", "reject"}


def test_regression_outputs_expected_metrics():
    size = 80
    frame = pd.DataFrame(
        {
            "text": [f"quality report {index}" for index in range(size)],
            "numeric": np.linspace(0, 5, size),
            "category": ["x", "y"] * (size // 2),
            "score": np.linspace(-2, 2, size),
        }
    )
    config = ExperimentConfig(
        target_column="score",
        task="regression",
        text_columns=("text",),
        tabular_columns=("numeric", "category"),
        model_names=("Ridge Regression",),
        transformer_model="fake",
        pca_mode="variance",
        pca_components=0.95,
    )
    result = run_experiment(frame, config, FakeEmbedder())

    assert {"r2", "rmse", "mae"} <= set(result.metrics.columns)
    assert len(result.predictions) == result.diagnostics["test_rows"]
    assert result.predictions["residual"].notna().all()


def test_sparse_high_cardinality_features_are_bounded():
    size = 600
    frame = pd.DataFrame(
        {
            "text": [f"record {index}" for index in range(size)],
            "category": [f"category_{index % 300}" for index in range(size)],
            "decision": ["yes", "no"] * (size // 2),
        }
    )
    config = ExperimentConfig(
        target_column="decision",
        task="classification",
        text_columns=("text",),
        tabular_columns=("category",),
        model_names=("Logistic Regression",),
        transformer_model="fake",
        pca_mode="none",
        retain_embeddings=True,
    )
    result = run_experiment(frame, config, FakeEmbedder())

    assert result.diagnostics["tabular_dimensions_after_encoding"] <= 256
    assert result.diagnostics["feature_matrix_format"] == "sparse CSR"
    assert result.embeddings is not None
    assert result.embedding_row_indices is not None
    assert len(result.diagnostics["dataset_fingerprint_sha256"]) == 64


def test_model_training_parameters_are_applied():
    logistic = _build_estimator(
        "Logistic Regression",
        "classification",
        42,
        {"C": 2.5, "max_iter": 700},
    )
    assert logistic.C == 2.5
    assert logistic.max_iter == 700

    forest = _build_estimator(
        "Random Forest",
        "regression",
        42,
        {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_leaf": 3,
            "max_features": 0.6,
        },
    )
    assert forest.n_estimators == 150
    assert forest.max_depth == 8
    assert forest.min_samples_leaf == 3
    assert forest.max_features == 0.6
