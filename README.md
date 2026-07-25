# Semantic Fusion Lab

A Streamlit application for multimodal supervised learning with structured
tabular variables and sentence-transformer text embeddings.

## Capabilities

- Upload CSV or Excel data.
- Validate uploaded structure, formulas, spreadsheet errors, infinities,
  missing values, zeros, duplicate rows, and constant columns before analysis.
- Select the dependent variable, natural-language columns, structured
  predictors, and an optional grouping variable.
- Run classification or regression.
- Choose a curated Sentence Transformers model or supply a compatible model ID.
- Apply no PCA, a fixed component count, or an explained-variance threshold.
- Compare Logistic/Ridge, XGBoost, CatBoost, and Random Forest models.
- Tune model-specific training parameters directly in the interface.
- Download metrics, test predictions, confusion matrices, configuration,
  diagnostics, and optional compressed embeddings.

Preprocessing and PCA are fitted only on training data. Group-aware splitting
is available for repeated entities such as papers, patients, or customers.
Sparse categorical encoding, bounded parallelism, and opt-in embedding
retention reduce memory pressure. See `ARCHITECTURE.md` for the scaling model
and production boundary.

## Local setup

Python 3.12 is recommended to match Streamlit Community Cloud's default.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first analysis downloads the selected sentence-transformer model. Later
runs reuse the local model cache.

## Dataset expectations

- One observation per row.
- One variable per column.
- A non-missing dependent variable for analyzed rows.
- At least one natural-language text column.
- At least one numeric, categorical, date, or identifier column as a structured
  predictor.
- For repeated entities, supply a group column and do not also use it as a
  predictor.

All data must be uploaded through the interface; the app does not load a
bundled or server-side dataset. Excel formula cells, spreadsheet errors,
infinite numeric values, empty columns, and invalid headers block analysis.
Zeros, missing values, duplicate rows, and constant columns trigger visible
warnings that must be reviewed and acknowledged. Text beginning with `=` must
be stored as a literal value rather than an Excel formula.

## Streamlit Community Cloud

After the code is pushed to GitHub:

1. Sign in at `share.streamlit.io`.
2. Create an app from the GitHub repository.
3. Select `streamlit_app.py` as the entrypoint.
4. Use Python 3.12 in Advanced settings.
5. Deploy.

Large transformer models consume more memory and have longer cold starts.
`all-MiniLM-L6-v2` is the lighter deployment option; `all-mpnet-base-v2`
matches the model used in the accompanying study.

The public demonstration is intentionally limited to 100 MB uploads, 50,000
rows, and 500 columns. Larger or concurrent production workloads should move
model execution to a queued worker service as described in `ARCHITECTURE.md`.

## Research note

This application supports exploratory supervised-learning experiments.
Credible conclusions still require appropriate sampling, group-aware or
temporal validation where relevant, hyperparameter tuning, robustness checks,
and external validation.
