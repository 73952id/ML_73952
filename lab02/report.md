# Lab 02 - A neural network from scratch

## 1. Goal

Implement a two-hidden-layer neural network using NumPy, check its analytical gradients, and compare learning rates and hidden-layer widths in terms of test accuracy and computation time.

## 2. Method

The implementation and saved execution outputs are in [neural_network.ipynb](neural_network.ipynb). The notebook contains the complete experiment in ten code cells.

### Dataset and network

The scikit-learn Digits dataset is scaled by dividing inputs by 16. Labels are one-hot encoded. The stratified 70/30 split uses `random_state=42`, producing 1,257 training samples and 540 test samples, each with 64 features.

The architecture is 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax, with W equal to 16 or 64. Weights have shapes (64,W), (W,W) and (W,10), initialized from N(0,0.01). The original implementation stores biases as row vectors of shapes (1,W), (1,W) and (1,10), allowing broadcasting over the batch. These differ in array shape from the one-dimensional biases listed in the manual but implement the same bias addition.

Softmax subtracts the largest logit in each row. Cross-entropy uses log-sum-exp for numerical stability. Parameter initialization and training shuffling each use a NumPy generator with seed 42; the data split also uses seed 42. The original notebook does not set Python or Torch global seeds and does not use those libraries for training.

### Backpropagation and gradient verification

For a batch of size B, the output gradient is dZ3=(P-Y)/B. Weight gradients are dW3=A2^T dZ3, dW2=A1^T dZ2 and dW1=X^T dZ1. Hidden gradients are dZ2=(dZ3 W3^T) * 1[Z2>0] and dZ1=(dZ2 W2^T) * 1[Z1>0]. Bias gradients sum the corresponding dZ over the batch, keeping a row dimension.

The forward-pass checks assert output shape (32,10) and probability row sums within 1e-6. The initial cross-entropy is printed for comparison with ln(10).

Gradient checking selects ten distinct random weights with selection seed 123 and perturbs each by +/-1e-5. Relative error is |analytical-numerical| / max(1e-8, |analytical|+|numerical|), with a passing threshold below 1e-6 for at least nine weights. A separate width-16 network initialized with seed 7 has its weights multiplied by ten to reduce cancellation in the numerical check. The training networks retain the prescribed standard deviation 0.01.

### Training and timing

All six combinations of widths {16,64} and learning rates {0.001,0.01,0.1} are trained for 50 epochs with shuffled mini-batches of 32. Each epoch prints loss and training accuracy in the notebook. Test accuracy is evaluated after training.

Forward and backward passes are timed separately on a batch of 32 for the learning-rate-0.1 models. Each measurement follows ten warm-up passes and reports the median of 100 runs. Backward timing uses a previously computed forward cache.

## 3. Results

The following values come from the original notebook's saved execution outputs.

| Check | Observed result |
|---|---|
| Output shape | (32,10) |
| Example probability row sum | 1.0000000000000002 |
| Initial cross-entropy | 2.302586 |
| Expected ln(10) | 2.302585 |
| Passing gradient checks | 10/10 |

Six sampled weights have zero analytical and numerical gradients. The four nonzero gradients have relative errors no larger than approximately 8.572e-9. Passing zero-gradient checks alone would not establish the correctness of all active paths; both kinds of samples are visible in the notebook output.

| Hidden width | Learning rate | Test accuracy |
|---:|---:|---:|
| 16 | 0.001 | 10.19% |
| 16 | 0.01 | 10.19% |
| 16 | 0.1 | 32.59% |
| 64 | 0.001 | 10.19% |
| 64 | 0.01 | 10.19% |
| 64 | 0.1 | 93.89% |

The loss curves are displayed in the notebook and saved by its plotting cell to `results/loss_curves.png`. At learning rate 0.1, the wider network learns substantially more within 50 epochs; smaller rates remain close to the untrained loss. The best model ends with training loss 0.1684 and training accuracy 94.67%.

| Hidden width | Forward median (ms) | Backward median (ms) |
|---:|---:|---:|
| 16 | 0.1368 | 0.1464 |
| 64 | 0.1946 | 0.2683 |

Backward is slower for both widths in the original saved run. Forward performs three matrix multiplications; backward performs five to compute weight gradients and propagate gradients through the hidden layers. This helps explain its larger cost, although allocations, activation masks, matrix shapes and library overhead also influence these small CPU timings.

## 4. Conclusions

1. Learning rate 0.1 produced much better results within 50 epochs than 0.001 or 0.01 with the required small weight initialization.
2. Increasing hidden width from 16 to 64 improved test accuracy from 32.59% to 93.89% at learning rate 0.1, while increasing the measured cost of both passes.
3. Numerical checks supported the implemented gradients, but correct gradients alone did not ensure that every learning-rate setting trained effectively.

## 5. Reproducibility and limitations

Run the notebook from `lab02/` so the relative output path points to this lab's results directory. Setup and kernel selection are described in the root [README](../README.md). The notebook is preserved with its original code and outputs; it does not call external experiment scripts.

The original saved session's complete dependency versions were not captured inside the assignment notebook. `results/versions.txt` records a later verification environment. The timing table above comes from the original saved notebook outputs; rerunning the notebook prints new measurements.

The prescribed settings were compared on the same test split; there is no separate validation split. Timings measure the development computer and can change between runs. The handwritten network design and gradient derivation are provided in [photo1](results/photo1.jpg), [photo2](results/photo2.jpg) and [photo3](results/photo3.jpg). The stored bias arrays use a leading singleton dimension, and the global seed calls specified by the manual are absent from the original notebook; these differences are documented rather than silently changing the student's code.
