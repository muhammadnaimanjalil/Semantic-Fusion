# Semantic Fusion Lab

Semantic Fusion Lab is a Streamlit application for supervised prediction using
a combination of structured tabular variables and natural-language text. It
converts selected text columns into sentence-transformer embeddings, combines
those embeddings with preprocessed tabular predictors, and trains classification
or regression models without requiring the user to write code.

The application is intended for research and exploratory predictive modelling
where potentially useful information is distributed across both structured
variables and unstructured text.

## Analytical objective

For each observation $i$, the uploaded dataset may contain:

- a dependent variable $y_i$;
- structured tabular predictors $x_i$; and
- one or more text fields $t_i$.

A sentence transformer converts the text into a dense semantic representation:

```math
z_i = f_{\theta}(t_i)
```

where $f_{\theta}$ is a pretrained sentence-transformer model and $z_i$ is
the resulting embedding vector.

After optional dimensionality reduction, the semantic features are concatenated
with the processed tabular features:

```math
h_i = [g(x_i), \tilde{z}_i]
```

where $g(x_i)$ represents the encoded and scaled tabular predictors and
$\tilde{z}_i$ represents either the original or PCA-reduced text embedding.

A supervised prediction model then estimates:

```math
\hat{y}_i = m(h_i)
```

For classification, $\hat{y}_i$ is a predicted class or class probability.
For regression, $\hat{y}_i$ is a predicted continuous value.

## Prediction and forecasting framework

The application performs held-out supervised prediction. It learns a
relationship between the selected independent variables and dependent variable
using a training subset, and evaluates that relationship on observations that
were not used to fit the model.

The application supports:

- **Classification**, where the dependent variable contains discrete classes
  such as accept/reject, positive/negative, or category labels.
- **Regression**, where the dependent variable is a continuous numeric outcome
  such as a score, quantity, rating, or future value.

In some research settings, this process may be described generally as
forecasting. However, the application does not automatically construct
time-series lags, rolling windows, or chronological validation periods.
Time-dependent studies require predictors and validation designs appropriate
to their temporal structure.

## Sentence-transformer representation

Traditional text models frequently represent documents through individual word
counts or sparse term-frequency features. Sentence transformers instead map
sentences or documents into dense vectors designed to preserve semantic
relationships.

Texts with related meanings may therefore have similar representations even
when they do not contain exactly the same words.

The user can:

- select one or more text columns;
- select a supported sentence-transformer model;
- provide a compatible custom model identifier;
- control the embedding batch size;
- optionally $L_2$-normalize the embeddings; and
- optionally retain the generated embeddings in the downloadable results.

When several text columns are selected, their values are combined while
preserving the identity of each source column.

## Dimensionality reduction with PCA

Sentence-transformer embeddings may contain hundreds of dimensions. Principal
Component Analysis (PCA) can optionally compress this semantic feature space
before it is combined with the tabular predictors.

PCA creates orthogonal linear combinations of the original embedding
dimensions that successively explain the greatest available variance. The
application supports:

- no PCA;
- a fixed number of principal components; or
- enough components to retain a selected proportion of explained variance.

PCA is fitted using training embeddings only. The fitted transformation is then
applied to the test embeddings. This prevents information from the test set
from influencing the representation learned during training.

PCA is applied only to sentence-transformer embeddings. Tabular variables are
handled by the separate tabular preprocessing pipeline.

## Tabular preprocessing

Selected structured predictors are divided into numeric and categorical
variables.

Numeric predictors are:

1. imputed using the training-set median; and
2. standardized using training-set means and standard deviations.

Categorical predictors are:

1. assigned an explicit missing-value category;
2. converted into one-hot encoded features; and
3. protected against previously unseen test-set categories.

Infrequent and high-cardinality categorical levels are constrained to reduce
memory consumption. The resulting tabular feature matrix is stored sparsely
where appropriate.

All preprocessing operations are fitted on training data only and then applied
to the held-out test data.

## Multimodal feature fusion

The application uses feature-level fusion. After tabular preprocessing and text
embedding generation, the two feature blocks are concatenated into a single
model input:

```math
X_{\text{multimodal}}
=
[X_{\text{tabular}}, X_{\text{text}}]
```

This design allows a prediction model to learn jointly from structured
attributes and semantic information contained in text.

The approach is model-agnostic: the same fused feature matrix can be supplied
to linear, bagging, or boosting estimators.

## Tabular-only baseline analysis

The user can optionally request a tabular-only baseline analysis.

For every selected prediction model, the baseline uses the same:

- dependent variable;
- tabular independent variables;
- analyzed observations;
- training and test split;
- group assignments;
- random seed;
- model hyperparameters; and
- evaluation metrics.

The only difference is that the text columns and sentence-transformer
embeddings are excluded:

```math
X_{\text{baseline}} = X_{\text{tabular}}
```

Comparing the multimodal and baseline results provides a direct assessment of
the incremental predictive value contributed by the semantic text features.
Because both analyses use the same held-out observations and model settings,
their performance measures are directly comparable within the experiment.

The baseline is not intended to prove that text has causal value. It measures
whether including the selected text representation improves out-of-sample
predictive performance under the configured experimental design.

## Supported prediction models

For classification, the application supports:

- Logistic Regression;
- Random Forest;
- XGBoost; and
- CatBoost.

For regression, it supports:

- Ridge Regression;
- Random Forest;
- XGBoost; and
- CatBoost.

Model-specific training parameters can be adjusted through the interface.
Depending on the model, these include regularization strength, number of trees
or boosting iterations, tree depth, learning rate, sampling fractions, minimum
leaf size, and related controls.

When several models are selected, they are trained and evaluated on the same
data split.

## Train/test splitting and leakage control

The analyzed observations are divided into training and test subsets.

For classification, the application can use stratified random splitting when
the class distribution permits it. Stratification attempts to preserve the
target-class distribution across the training and test sets.

A group column can be selected when several rows belong to the same underlying
entity, such as:

- multiple reviews of the same paper;
- repeated observations from the same patient;
- transactions from the same customer; or
- records from the same organization.

With group-aware splitting, all rows belonging to a group are assigned to
either training or testing, but never both. This reduces information leakage
caused by the same entity appearing on both sides of the evaluation.

## Evaluation metrics

Classification results include:

- accuracy;
- weighted $F_1$ score;
- macro $F_1$ score;
- weighted precision;
- weighted recall; and
- confusion matrices where the number of classes is manageable.

Regression results include:

- coefficient of determination ($R^2$);
- root mean squared error (RMSE); and
- mean absolute error (MAE).

Predictions are calculated only for the held-out test observations.

When the baseline option is enabled, the results identify each row as either
**Multimodal** or **Tabular baseline**, allowing model-level comparisons between
the two feature configurations.

## Analytical pipeline

The application implements the following workflow:

1. **Upload data**

   Upload a CSV or Excel dataset and select the appropriate worksheet.

2. **Validate the dataset**

   Inspect the table for structural problems, formulas, spreadsheet errors,
   infinite values, zeros, missing values, duplicate rows, and constant columns.

3. **Assign variable roles**

   Select the dependent variable, text predictors, tabular predictors, and an
   optional grouping variable.

4. **Choose the prediction task**

   Configure either classification or regression.

5. **Generate semantic text features**

   Convert the selected text into dense sentence-transformer embeddings.

6. **Apply optional PCA**

   Fit PCA on the training embeddings and transform both training and test
   embeddings.

7. **Preprocess tabular predictors**

   Impute, scale, and encode variables using transformations learned only from
   the training data.

8. **Construct the multimodal feature matrix**

   Concatenate the processed tabular features and text embeddings.

9. **Construct the optional baseline matrix**

   Retain only the processed tabular features while keeping all other
   experimental settings unchanged.

10. **Train prediction models**

    Fit each selected estimator using the configured hyperparameters.

11. **Evaluate held-out performance**

    Generate test-set predictions, metrics, and confusion matrices.

12. **Download reproducibility outputs**

    Export metrics, predictions, configuration details, diagnostics, and
    optional embeddings.

## Data-quality checks

The application distinguishes between blocking errors and reviewable warnings.

Blocking issues include:

- empty datasets;
- invalid or duplicate column headers;
- columns containing no values;
- positive or negative infinity;
- spreadsheet error values; and
- Excel formula cells.

Warnings include:

- numeric zero values;
- missing cells;
- duplicate rows; and
- constant columns.

Zeros are not automatically treated as errors because zero may be a valid
measurement or category code. The application requires the user to review and
acknowledge warnings before proceeding.

Rows with missing dependent-variable values are excluded from the analysis.
Missing predictor values are handled by the training-only preprocessing
pipeline.

## Dataset expectations

The uploaded dataset should contain:

- one observation per row;
- one variable per column;
- meaningful and unique column names;
- a dependent variable with sufficient non-missing observations;
- at least one natural-language text column; and
- at least one structured tabular predictor.

For classification, the dependent variable must contain at least two classes.
Rare classes must have enough observations or groups to support a valid
training and test split.

For repeated entities, a group identifier should be selected and should not
also be used as a predictor.

All data must be uploaded through the application interface. The application
does not automatically use bundled or server-side datasets.

## Capabilities

- Upload CSV, XLSX, or XLSM data.
- Select an Excel worksheet.
- Validate data before analysis.
- Configure classification or regression.
- Select multiple text and tabular predictors.
- Use group-aware leakage-resistant splitting.
- Choose from curated or compatible custom sentence transformers.
- Apply optional embedding normalization and PCA.
- Compare multiple prediction models.
- Adjust model-specific hyperparameters.
- Run an optional tabular-only baseline.
- Inspect metrics, predictions, diagnostics, and confusion matrices.
- Download a reproducibility bundle.
- Optionally retain compressed embedding vectors.

## Downloadable results

The results ZIP file can contain:

- `metrics.csv`;
- `predictions.csv`;
- `configuration.json`;
- `diagnostics.json`;
- model-specific confusion matrices; and
- optional compressed sentence embeddings.

The configuration records the selected variables, transformer model, PCA
settings, split settings, prediction models, model hyperparameters, and whether
the tabular baseline was enabled.

Diagnostics include data dimensions, train/test indices, split method,
embedding dimensions, encoded tabular dimensions, feature-matrix information,
software versions, and a dataset fingerprint.

## Local setup

Python 3.12 is recommended because it matches the automated test environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first analysis may download the selected sentence-transformer model.
Subsequent runs can reuse the local model cache.

Run the automated checks with:

```powershell
python -m pip install -r requirements-test.txt
ruff check data_validation.py multimodal_pipeline.py streamlit_app.py tests
pytest -q
```

## Scalability and software design

The analytical pipeline is separated from the Streamlit interface so that it
can be tested independently and reused from another interface or notebook.

The application uses:

- cached sentence-transformer resources;
- batched text encoding;
- sparse categorical features;
- bounded model parallelism;
- limits on upload size, rows, and columns;
- optional rather than mandatory embedding retention;
- deterministic random seeds; and
- automated linting and tests.

Large datasets, large transformer models, and simultaneous users may require a
dedicated model-serving or queued-worker architecture. See `ARCHITECTURE.md`
for additional implementation and scaling details.

## Interpretation and research limitations

Performance estimates depend on data quality, sample size, variable selection,
class balance, group structure, train/test design, model settings, and domain
validity.

A strong held-out result does not by itself demonstrate:

- a causal relationship;
- generalization to different populations or periods;
- robustness to distributional change;
- absence of bias;
- appropriate probability calibration; or
- suitability for consequential automated decision-making.

Credible research conclusions may additionally require temporal or external
validation, repeated cross-validation, hyperparameter-selection protocols,
robustness and sensitivity analyses, fairness assessment, and comparison with
domain-specific benchmarks.
