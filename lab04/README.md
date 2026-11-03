# Lab 04: Data pipeline and quality

Reproducible Adult data preparation and Logistic Regression comparison. The notebook follows manual sections 1-7 and includes saved outputs.

## Files

```text
lab04/
  README.md
  data_pipeline.ipynb
  src/
    __init__.py
    transforms.py
  results/
  data_card.md
  model_card.md
```

- `data_pipeline.ipynb`: problem definition, ingestion, validation, preprocessing, feature comparison, label audit and reproducibility artifacts.
- `src/transforms.py`: train-fitted imputation and clipping, plus derived features.
- `data_card.md`: data source, license, quality, biases and privacy.
- `model_card.md`: model configuration, evaluation and limitations.

## Run

Use the shared course environment from the repository's requirements, with PyYAML 6.0.3. Open the notebook from the repository root or `lab04/` and run its cells in order. The first run needs network access for OpenML; later runs use `data/openml/` at the repository root.

One command from the repository root:

```powershell
.\lab01\.venv\Scripts\python.exe -c "import json; n=json.load(open('lab04/data_pipeline.ipynb',encoding='utf-8')); [exec(compile(''.join(c['source']),'<notebook>','exec'),globals()) for c in n['cells'] if c['cell_type']=='code']"
```

## Results

| File | Contents |
|---|---|
| `validation_report.json` | Schema, missingness, duplicate and label-quality summaries |
| `violations.csv` | Invalid types, categories or ranges; empty apart from headers in this run |
| `label_conflicts.csv` | Five identical-feature groups with conflicting labels |
| `review_50.csv` | Sampled source rows, review statuses and notes; retained for unchanged records on reruns |
| `metrics.csv` | Baseline and engineered-feature test accuracy and F1 |
| `preprocessor.joblib` | Fitted engineered preprocessor; the classifier itself is not saved |
| `processed.csv` | Imputed, clipped and engineered data before scaling/encoding, with row ID, split and label |
| `processed.sha256` | SHA256 of the processed CSV bytes |
| `config.yaml` | Configuration, library versions and data hashes |

Load the saved preprocessor with `lab04/` on Python's import path so that `src.transforms` is importable.
