# Machine Learning Systems - 73952

Laboratory work for the Machine Learning Systems course. The labs can be opened and run as Jupyter notebooks. The datasets are included with scikit-learn; no separate download is needed.

## Notebooks

- [Lab 01: environment and first system measurements](lab01/experiment.ipynb): package versions, Logistic Regression and Random Forest experiments.
- [Lab 02: a neural network from scratch](lab02/neural_network.ipynb): the original completed notebook, including all ten code cells and their saved outputs. It contains the dataset preparation, initialization, forward pass, loss, backpropagation, gradient check, training, plots and timing measurements.
- [Lab 02: package versions](lab02/print_versions.ipynb): a separate environment check; this is not the neural-network assignment.

- [Lab 03: architectures and their costs](lab03/architectures.ipynb): PyTorch MLP, CNN and patch attention on Fashion-MNIST, parameter/FLOP comparison, five-epoch training and latency benchmarks.

Reports: [Lab 01](lab01/report.md), [Lab 02](lab02/report.md), [Lab 03](lab03/report.md).

## Environment

Use Python 3.11.8. Shared pinned dependencies are in `requirements.txt`. From the repository root in PowerShell, the existing environment can be activated with:

```powershell
.\lab01\.venv\Scripts\Activate.ps1
python --version
python -m pip check
```

This environment was checked against the shared requirements. The separate `lab02/.venv` has different library versions. To create a fresh shared environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

Check that `python --version` is exactly 3.11.8; `py -3.11` selects the minor version, not necessarily that patch release.

The manual lists `pyarrow==16.1.0`, but `mlflow==2.14.1` requires `pyarrow<16`. The working requirements retain `pyarrow==15.0.2`; this exception needs instructor approval.

## Running in Jupyter / VS Code

Open the notebook and select the Python interpreter from `lab01/.venv` (or the newly installed root `.venv`) as its kernel. Run cells from top to bottom. If VS Code requests a kernel dependency, install it in that selected environment.

For Lab 01, use `lab01/` as the notebook working directory; the first cell also supports execution from the repository root. For Lab 02, use `lab02/` as the working directory: the original notebook saves `results/loss_curves.png` relative to the current directory. The original Lab 02 cells are preserved without wrappers or changes to their algorithm.

For a standalone Jupyter interface, install the optional notebook tools in the selected environment and start from the lab directory:

```powershell
python -m pip install jupyterlab ipykernel
Set-Location lab02
python -m jupyter lab neural_network.ipynb
```

From that directory, Lab 01 is available at `../lab01/experiment.ipynb`.

## Results and reproducibility

Lab 01 measurements and model files are in `lab01/results/`. Lab 02's original saved outputs are inside `neural_network.ipynb`; its plotting cell writes `lab02/results/loss_curves.png` when run from `lab02/`.

The Lab 02 report uses the original notebook outputs, including its timing measurements. `lab02/results/versions.txt` records the environment used for a later verification run, not a verified record of the original saved notebook session. Handwritten work is stored in `lab02/results/photo1.jpg`, `photo2.jpg` and `photo3.jpg`.

Lab 03 downloads Fashion-MNIST automatically into the ignored root `data/` directory. Its notebook supports execution from the repository root or `lab03/` and saves experiment tables, plots and library versions in `lab03/results/`. Run all cells with the same course kernel. CPU execution is supported; the measured device and thread count are recorded in `lab03/results/versions.txt`.
