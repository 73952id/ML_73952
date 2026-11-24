# Model card

**Purpose:** suggest one of eight music genres and mark uncertain predictions for human review. Author: Mariia Liutenko, 73952.

**Architecture:** three convolution blocks (16, 32 and 64 channels), BatchNorm, ReLU, two max pools, global average pooling and an eight-class linear classifier. There are 24,040 trainable parameters. Train-only input normalization is stored in model buffers. The deployed artifact is a CPU TorchScript model.

**Training:** 400 FMA-small pilot tracks, one middle five-second segment per track, Adam with learning rate 0.001, batch size 32, cross-entropy, 20 epochs and seed 42. The checkpoint is selected by macro-F1 on 80 validation tracks. Evaluation uses 80 separate test tracks.

**Decision rule:** softmax gives genre scores; the largest score selects the suggested genre. A validation-selected threshold controls automatic tagging. The selected threshold maximizes coverage among thresholds reaching at least 80% accepted validation accuracy with at least five accepted examples. If none exists, the threshold is set above 1, disabling automatic tagging. Test results determine whether the actual proposal target of at least 80% accepted accuracy and 50% coverage was reached.

**Measurements:** `results/metrics.json` reports accuracy, macro-F1, accepted-tag accuracy and coverage, serialized bytes, warm CPU forward p50/p95 and total pipeline p95. The forward timing uses batch size one, two threads, 20 warm-ups and 200 measured calls on the saved, reloaded model. `results/timings.csv` separates preprocessing from inference.

**Limits:** this is a pilot, not a full FMA-small result. Genre boundaries are subjective. A high score can still be wrong, especially for music outside the eight supported classes. Scores are not calibrated correctness probabilities. CSV tags are suggestions; the app does not edit original audio metadata. Full-data training and INT8 optimization remain future work.
