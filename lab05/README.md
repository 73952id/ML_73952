# Lab 05: Frameworks and automatic differentiation

`autodiff.ipynb` follows manual sections 1-6 and stores executed outputs. Source code is under `src/`; measurements and graph inspection outputs are under `results/`.

## Run

Use the shared course Python environment specified in the repository's `requirements.txt`. Open the notebook and run cells in order from the repository root or `lab05/`. A standalone runner also executes the same notebook and saves its outputs:

```powershell
.\lab01\.venv\Scripts\python.exe lab05/src/run_notebook.py
```

The experiments use CPU, float64 and two PyTorch threads. Training compares the median of five matched runs. Execution timings use TorchScript tracing and freezing with separate warm-up and setup timing.

## Files

- `src/value.py`: scalar nodes, operations, topological ordering and reverse-mode differentiation.
- `src/experiments.py`: gradient checks, make_moons model training and PyTorch execution/layout benchmarks.
- `src/run_notebook.py`: notebook execution and output persistence.
- `report.md`: methods, measured results and interpretation.
- `data_card.md`: generated data and split.
- `model_card.md`: architecture, training and limitations.

## Results

| File | Contents |
|---|---|
| `gradient_check.csv` | Custom, PyTorch and numerical gradients and absolute errors |
| `training_runs.csv` | Five training times and evaluation scores per implementation |
| `training_medians.csv` | Median comparison |
| `training_history.csv` | BCE over epochs for the first matched run |
| `config.json` | Data generation, split IDs, scaling and training settings |
| `traced_graph.txt` | Frozen, inlined TorchScript inference graph |
| `execution_timings.csv` | Warm eager/traced timings, trace setup and estimated break-even |
| `tensor_layout.csv` | Shape, dtype, strides, contiguity and matmul times |
| `layout_copy_cost.json` | Time to create the contiguous copy |
| `versions.json` | Runtime versions and device settings |

Rerunning replaces numerical measurements. Trained weights are held in the notebook session; no checkpoint is exported.
