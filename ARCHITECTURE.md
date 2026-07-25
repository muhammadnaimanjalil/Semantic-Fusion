# Architecture and scalability

## Components

- `streamlit_app.py` owns file ingestion, configuration controls, model caching,
  presentation, and safe result export.
- `multimodal_pipeline.py` is framework-independent. It validates the requested
  experiment, creates a leakage-aware split, prepares features, fits models,
  and returns structured results.
- `tests/` exercises the reusable pipeline with deterministic synthetic
  embeddings, so CI does not download transformer models.

## Data flow

1. Read one CSV or Excel worksheet.
2. Validate row, column, target, predictor, and split constraints.
3. Keep grouped entities in one split when a group column is selected.
4. Encode selected text columns in batches with a shared transformer instance.
5. Fit tabular preprocessing and PCA on training rows only.
6. Combine sparse tabular features with dense float32 embeddings.
7. Fit each selected estimator and evaluate only on held-out rows.
8. Export metrics, predictions, configuration, diagnostics, and optional
   compressed embeddings.

## Current scalability controls

- A transformer model is cached once per application process and shared across
  sessions.
- A lock serializes CPU embedding calls to avoid multi-session memory spikes.
- Categorical features use sparse one-hot encoding, group infrequent levels,
  and cap each selected categorical column at 256 output categories.
- Parallel estimators use at most four CPU workers.
- Embeddings are retained in session memory only when explicitly requested.
- Probability columns and confusion matrices are capped for high-cardinality
  classification targets.
- Uploads are limited to 100 MB, 50,000 rows, and 500 columns.
- Dataset fingerprints, split seeds, package versions, and configuration are
  included in the downloadable diagnostics.

## Deployment boundary

The single-process Streamlit design is appropriate for demonstrations,
teaching, research prototypes, and moderate CPU workloads. Streamlit Community
Cloud has finite shared resources, so large datasets and large transformer
models will have slow cold starts and can exceed memory limits.

For production-scale or multi-user workloads, retain the current pipeline API
but move model execution to a separate service:

1. Store uploads and outputs in managed object storage.
2. Submit experiments to a durable task queue.
3. Run embedding and model jobs on autoscaled CPU/GPU workers.
4. Persist job status and metadata in a database.
5. Let Streamlit poll job status and retrieve completed artifacts.

This preserves the interface while allowing independent scaling, retries,
quotas, authentication, audit logs, and data-retention controls.
