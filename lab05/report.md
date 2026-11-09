# Lab 05: Frameworks and automatic differentiation

## 1. Value node

`src/value.py` defines a scalar Value with data, a zero-initialized gradient, parent references and a local derivative rule. Parent tuples retain repeated operands. Leaf nodes have no parents and a no-op local rule.

## 2. Operations and graph construction

Addition, multiplication, constant powers and ReLU create new nodes during the forward pass. Their local derivatives are 1, the opposite operand, p*x**(p-1), and the positive-input indicator, respectively. ReLU uses derivative zero at zero. Local rules multiply these derivatives by the upstream gradient and accumulate with +=. Sigmoid, log, subtraction and division support mean BCE training. Scalar sum is an addition of zero, allowing the prescribed expression to be evaluated with the scalar engine.

The notebook constructs ReLU((a*b+c)**2).sum() for a=2, b=-3, c=1. Its output is 25.

## 3. Backward pass

An iterative depth-first traversal produces a topological order with parents before children. Every node is visited once. The output gradient is set to 1, and local rules run in reverse order. For the example, gradients are a=30, b=-20, c=-10. A shared-subgraph check, x*x + x*x + x with the product node reused, gives 13 at x=3, demonstrating accumulation across branches. Leaf parameter gradients are cleared before each training step; the forward graph is rebuilt each step.

## 4. Gradient verification

The required expression is compared against float64 PyTorch autograd and central finite differences (step 1e-5) for four input triples. Maximum absolute custom-versus-PyTorch error: **0**. Maximum custom-versus-numerical error: **6.40622e-10**. Both are below the required 1e-6. Negative inputs, a zero result and ReLU at zero are checked. All 33 initial model parameter gradients also match PyTorch within **2.08167e-17**. Full results are in `results/gradient_check.csv`.

## 5. make_moons training and speed

Generate 100 samples with noise 0.2 and seed 42; use an 80/20 stratified train/test split. StandardScaler is fitted only on train. Both models have architecture 2-8-1 with sigmoid hidden and output activations, 33 parameters, mean BCE, float64 and full-batch SGD (learning rate 1.0, 200 epochs). Initial parameters and data are identical in all five fresh runs. Final parameter arrays agree within 1e-6 after each matched run.

| Implementation | Median training seconds (5 runs) | Last pre-update training BCE | Test accuracy |
|---|---:|---:|---:|
| Scalar Value engine | 30.590980 | 0.285802 | 85.00% |
| PyTorch | 0.332557 | 0.285802 | 85.00% |

PyTorch is about **91.99 times faster** in this measured training loop. Timing includes forward, loss, backward, gradient clearing and SGD updates, including scalar Python graph allocation and garbage collection; it excludes model/optimizer setup and evaluation. The reported BCE is from the last forward pass before its update; test accuracy is evaluated after that update. First-run histories are in `training_history.csv` and all ten timings are in `training_runs.csv`.

The scalar implementation performs individual operations in Python, while PyTorch batches the calculation into tensor kernels. The comparison measures these implementations, not automatic differentiation in isolation. All repetitions use the same dataset and initialization; they measure timing variation rather than accuracy uncertainty. Test accuracy uses only 20 held-out examples.

## 6. PyTorch graph, execution modes and tensor layout

The trained Sequential model is traced with torch.jit.trace and frozen for inference. The inlined graph in `results/traced_graph.txt` contains two aten::linear and two aten::sigmoid operations, with model weights/biases embedded as constants. Traced outputs match eager outputs (maximum error 0). Tracing/freezing setup took **314.929 ms**, excluded from warm inference timings.

| Batch | Eager median ms/call | Traced median ms/call | Speedup | Approximate calls to amortize setup |
|---|---:|---:|---:|---:|
| 1 | 0.129153 | 0.083909 | 1.539x | 6961 |
| 128 | 0.152355 | 0.113674 | 1.340x | 8142 |

Each median uses 30 timed groups of 100 calls after 20 warm-up calls. Both paths run in eval/inference mode on CPU with two threads. Graph capture helps repeated calls to a stable model by reducing Python/module dispatch overhead. For only a few calls, setup can exceed the time saved. These measurements use tracing/freezing; they do not establish torch.compile performance or behavior for input-dependent Python branches. The estimated break-even is setup time divided by the measured per-call saving and assumes stable timing.

A 512x512 float64 transpose view has strides (1,512), is non-contiguous and shares storage with its base. Its contiguous copy has strides (512,1), does not share storage and contains exactly the same values. Both are multiplied by the same 512x128 float64 matrix, with matching outputs. Timing excludes copy creation and uses 30 groups of 20 calls after 20 warm-ups.

| Input | Median matmul ms |
|---|---:|
| Non-contiguous transpose | 3.869362 |
| Contiguous copy | 3.728022 |

The transpose operation was **1.038 times** the contiguous matmul time in this run, a small difference. Creating the contiguous copy separately cost **1.474725 ms**, so copying for a single operation is not automatically beneficial. Matrix multiplication libraries can handle strided/transposed inputs efficiently; contiguity is not a universal speed guarantee. Results depend on the operation, dimensions, CPU and system load. Runtime versions and thread settings are saved in `results/versions.json`.

## Sources

Assignment: supplied Lab Manual: Labs 05, Frameworks and automatic differentiation. Reading: Reddi, Chapter 7, sections 7.3, 7.4, 7.6 and 7.7-7.8. Experiment code is in `autodiff.ipynb` and `src/`; saved tables contain the measured evidence.
