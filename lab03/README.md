# Lab 03: Architectures and their costs

Comparison of MLP, CNN and patch-attention classifiers on Fashion-MNIST: accuracy, parameters, FLOPs and inference latency.

## Files

```text
lab03/
  README.md
  architectures.ipynb
  src/
    __init__.py
    models.py
  results/
  data_card.md
  model_card.md
  report.md
```

- `architectures.ipynb`: dataset preparation, model inspection, profiling, training, latency measurement and plots, with saved outputs.
- `src/models.py`: MLP, CNN and PatchAttention definitions.
- `report.md`: methods, hand calculations, measured results and conclusions.
- `data_card.md`: dataset provenance, preprocessing and limitations.
- `model_card.md`: model configurations, evaluation and limitations.

## Run

Use the shared course environment from the repository's `requirements.txt`. Open the notebook from the repository root or `lab03/` and run all cells in order. Fashion-MNIST is downloaded to the ignored repository-level `data/` directory on the first run. CPU is supported; CUDA is used if available.

## Results

| File | Contents |
|---|---|
| `measurements.csv` | Parameters, FLOPs, test accuracy and median inference latency |
| `model_summaries.txt` | torchinfo model summaries |
| `training_history.csv` | Train/validation loss and accuracy for five epochs |
| `tradeoffs.png` | Accuracy versus parameters and batch-1 latency |
| `validation_curves.png` | Validation accuracy over epochs |
| `versions.txt` | Environment versions, device and thread settings for the recorded run |
| `photo1.jpg` - `photo4.jpg` | Handwritten layer FLOP calculations |

Saved results describe the recorded CPU run. Re-running training and timing can change measurements; no trained weight checkpoints are stored.
