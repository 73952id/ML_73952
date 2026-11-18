# Lab 07: Compression - pruning and distillation

`compression.ipynb` follows the five manual sections and contains executed outputs and plots. Models and experiment code are in `src/`; measured results and serialized models are in `results/`.

## Run

Use the shared course environment from the repository's `requirements.txt`. From the repository root:

```powershell
.\lab01\.venv\Scripts\python.exe lab07/src/run_notebook.py
```

Alternatively open `compression.ipynb` from the root or `lab07/` and run its cells in order. Fashion-MNIST is loaded from the ignored root `data/` cache. Training uses the complete 55,000-image training partition, CPU, float32 and two Torch threads. Completed variants are reused only when the saved configuration and source hash match. To repeat timing/training from scratch, preserve the old results separately and start with an empty `lab07/results/` directory.

## Files

```text
lab07/
  README.md
  compression.ipynb
  report.md
  data_card.md
  model_card.md
  src/
    models.py
    experiment.py
    run_notebook.py
  results/
```

- `src/models.py`: CNN, exact-count magnitude masks and physical filter removal.
- `src/experiment.py`: training, validation, storage round-trip checks, latency and efficiency tables.
- `src/latency.py`: post-training latency rounds in shuffled model order.
- `src/report.py`: builds the report from recorded measurements.
- `src/run_notebook.py`: executes the same notebook and saves its real outputs, including figures.

## Results

| File | Contents |
|---|---|
| `measurements.csv` | Every model variant: accuracy, storage, parameters, FLOPs, latency and CPU time |
| `training_history.csv` | Training loss, CE/KL components and validation metrics per epoch |
| `sparsity_curves.png` | Global/layer-wise weight sparsity and physical filter-pruning curves |
| `storage_latency.png` | Dense CPU latency and dense/CSR file sizes across sparsities |
| `efficiency_pillars.csv` | Consolidated data/model/compute/energy-related comparison |
| `latency_repetitions.csv` | Three post-training latency rounds in shuffled model order; final latency is their median |
| `initial_measurements.csv` | Original per-variant measurements before the interleaved timing check |
| `config.json` | Experiment settings, preprocessing and source hash |
| `versions.json` | Runtime and hardware/thread metadata |
| `models/*.pt` | Dense checkpoints with architecture and state dictionary |
| `models/*_csr.npz` | Lossless uncompressed CSR arrays and architecture metadata |

The global and layer-wise 0% cases use one identical three-epoch control; the alias and original artifact paths are recorded. Sparse archives are storage representations, not a sparse inference backend. Timed inference always uses dense PyTorch kernels. Final latency uses three post-training rounds in shuffled order. Hardware energy is unavailable: CPU time is recorded as an activity proxy and is not converted into Joules. Training times include per-epoch validation. The runner also regenerates report.md from saved measurements.
