# Scalar and PyTorch MLP model card

## Task and architecture

Binary classification of synthetic make_moons coordinates. Both implementations have 2 inputs, 8 sigmoid hidden units and 1 sigmoid output, with 33 trainable scalar parameters. Predictions use probability >=0.5 for class 1.

## Training

Mean binary cross-entropy, full-batch SGD, learning rate 1.0 and 200 epochs. Float64 precision, identical initial weights and biases, and the same scaled 80-point training partition are used. Initial weight matrices are drawn with NumPy default_rng(42) from a normal distribution with standard deviation 0.5; biases start at zero. Each of five repetitions starts from those same weights. Parameter gradients are zeroed before each step.

## Validation and timing

The required expression is checked against PyTorch autograd and central finite differences; both maximum errors must be below 1e-6. All 33 initial model parameter gradients and final matched-run weights are also checked against PyTorch. Test accuracy is measured on 20 held-out points.

Training timings include forward, BCE, backward and SGD steps and exclude model/data setup and evaluation. Medians of five runs are in `results/training_medians.csv`; individual runs are in `training_runs.csv`. The speed comparison reflects a Python scalar graph versus vectorized PyTorch on CPU, including implementation overhead and automatic garbage collection.

The trained PyTorch model is used for eager versus frozen TorchScript tracing experiments. These are inference-only CPU measurements after warm-up. Trace setup is measured separately; gradients and training behavior are not benchmarked through the frozen graph.

## Limitations and artifacts

Only one small generated dataset and one initialization are used. Repetitions measure timing variation, not statistical accuracy variation. CPU timing is environment-dependent. TorchScript tracing records the observed operations; it does not prove correctness for arbitrary input-dependent Python control flow. This Sequential model has no such branches. See `report.md` for actual measurements and interpretation. No trained checkpoint is saved.
