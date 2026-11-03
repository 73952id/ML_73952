# Adult income classifier model card

## Task and intended use

Binary classification of annual income above $50,000 using Adult version 2 from OpenML. Intended for the Lab 04 educational comparison of preprocessing and feature engineering. Inputs are 14 census attributes; the engineered variant adds `capital_net` and `hours_per_age`. The positive label is `>50K`.

## Training and preprocessing

Logistic Regression with `solver=lbfgs`, `C=1.0`, `max_iter=3000` and seed 42. Both variants use the same stratified 70/30 split. Training has 34,165 rows after removing 24 exact duplicates; test retains all 14,653 rows.

Imputation, 1st/99th percentile clipping, scaling and categorical encoding are fitted on train only. Numeric features use median imputation and StandardScaler; categorical features use mode imputation and OneHotEncoder with unknown categories ignored. Derived features use the imputed, clipped values. Prediction uses the default binary decision threshold of 0.5. No hyperparameter search is performed.

## Evaluation

| Variant | Test accuracy | Test F1 (>50K) | Solver iterations |
|---|---:|---:|---:|
| Original features | 0.853955 | 0.660102 | 61 |
| Plus two derived features | 0.857777 | 0.669626 | 75 |

Both fits converged. The engineered variant improves accuracy by about 0.38 percentage points and F1 by 0.0095 on this split. These are single-split results; no statistical significance claim or separate attribution to either derived feature is made.

## Limitations

The historical, imbalanced dataset contains demographic and income biases. No subgroup fairness evaluation is performed. Row-wise splitting and train-only exact deduplication do not ensure that feature-identical records are separated by group. Five feature-identical groups have conflicting income labels. Aggregate accuracy and F1 do not establish suitability for decisions about individuals or causal relationships.

## Artifacts and reproducibility

`data_pipeline.ipynb` contains the training and evaluation code. `results/metrics.csv` stores the measured scores; `results/config.yaml` records settings and versions. `results/preprocessor.joblib` stores the engineered preprocessor, not the trained classifier. Re-run the notebook to train the classifiers. See `data_card.md` for data provenance and licensing.
