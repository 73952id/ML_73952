# Fashion-MNIST classifiers model card

## Task and architectures

Ten-class classification of 28 x 28 grayscale clothing images. Models are defined in `src/models.py`; all output ten logits.

- MLP: Flatten, Linear 784-256, ReLU, Linear 256-128, ReLU, Linear 128-10.
- CNN: Conv 1-16 with 3 x 3 kernel and padding 1, ReLU, MaxPool 2; Conv 16-32 with the same kernel/padding, ReLU, MaxPool 2; Flatten, Linear 1568-10.
- Attention: sixteen 7 x 7 patches, shared Linear 49-64 projection, two Transformer encoder layers (dimension 64, four heads, feedforward width 128), mean pooling and Linear 64-10. PyTorch defaults include ReLU, dropout 0.1 and LayerNorm. No positional embeddings are added.

## Training and evaluation

Each model uses Adam with learning rate 0.001, batch size 128 and five epochs. Cross-entropy is the loss. Seed 42 controls initialization, the 55,000/5,000 train/validation split and batch shuffling. The final fifth-epoch model is evaluated on the separate official 10,000-image test set.

| Model | Parameters | Test accuracy | Batch 1 median ms | Batch 128 median ms |
|---|---:|---:|---:|---:|
| MLP | 235,146 | 87.64% | 0.4825 | 2.8332 |
| CNN | 20,490 | 89.17% | 1.0663 | 38.0417 |
| Attention | 70,794 | 84.24% | 3.3568 | 26.1463 |

Timing uses prepared tensors, eval/inference mode, one warm-up and the median of 100 runs for batch 1 or 20 for batch 128. The recorded run uses CPU, two Torch threads and disabled MHA fastpath. Data loading and device transfers are excluded.

## Computational cost

Hand matrix-product counts are 469,504 FLOPs/image for MLP, 2,063,488 for CNN and 2,329,856 for Attention, counting multiply-add as two operations. A custom attention hook supplements THOP. Bias, activation, softmax and other scalar operations are not fully represented; details and gaps are explained in `report.md`.

## Limitations and intended use

This is a single-seed, five-epoch educational comparison. Prescribed architectures do not meet the manual's suggested 100k-150k parameter range and are not matched for parameter count or compute. Timing is hardware/backend-specific. Position-free attention with mean pooling is invariant to whole-patch permutations. Test accuracy does not establish real-world clothing recognition performance. CNN has the highest recorded accuracy and fewest parameters; MLP has the lowest measured batch-1 latency.

## Artifacts

The notebook contains training and evaluation code with saved outputs. `results/measurements.csv` and `training_history.csv` contain measurements; `versions.txt` records the measured environment. No trained weights are saved. See `data_card.md` for data provenance and `report.md` for analysis.
