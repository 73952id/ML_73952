# Fashion-MNIST compressed CNN model card

## Models

Teacher: Conv(1,8,3,padding=1)-ReLU-Pool2, Conv(8,16,3,padding=1)-ReLU-Pool2, Flatten, Linear(784,1280)-ReLU-Linear(1280,10). Total: 1,018,858 parameters. Student uses hidden width 128, for 103,018 parameters. Outputs are ten logits.

Global and layer-wise variants zero the smallest absolute conv/linear weights at requested sparsities 0, 25, 50, 75, 90 and 95 percent. Biases are not pruned. Gradient masking and post-update projection keep masks fixed. Structured variants physically remove convolution output filters ranked by L2 norm, adjusting subsequent input channels and fully connected columns. Requested filter removals are 10, 25 and 50 percent, rounded to integer channel counts; actual fractions are recorded.

## Training

Adam, batch size 512 and seed 42. Teacher and both students train for five epochs at learning rate 0.001. Every pruned variant receives three fine-tuning epochs at 0.0001 from the same original teacher. The two zero-sparsity procedures share their identical fine-tuned control. Student initialization and batches match between plain and distilled runs.

Distillation uses T=4 and alpha=0.5, with the manual's KL(student || teacher) direction: alpha*CE + (1-alpha)*T^2*KL. The teacher is frozen and evaluated without gradients. Distillation includes teacher forward compute in its recorded training cost.

## Evaluation and artifacts

Final checkpoints are evaluated on all 10,000 official test images. `results/measurements.csv` and `efficiency_pillars.csv` contain the recorded accuracy, parameter counts, file bytes, dense/ideal-nonzero FLOPs and latency. `report.md` explains the results. Dense architecture/state checkpoints and exact CSR storage archives are in `results/models/`.

Each inference timing round uses 20 warm-ups and the median of 100 runs for batch 1; batch 128 uses five warm-ups and 20 runs. Final latency is the median of three post-training rounds in shuffled model order, with individual results saved. Timings exclude data loading and storage conversion. All models run on CPU with two threads. Sparse archives are not timed as sparse inference. Training times include per-epoch validation.

## Limits

Unstructured zeros retain dense tensor shapes and nominal executed compute. Ideal nonzero operation counts are hypothetical sparse-kernel work, not measured operations or speedups. CSR indices can increase storage at low sparsity. Accuracy/latency differences are single-seed observations without confidence intervals. Teacher and pruned variants have unequal cumulative training budgets; the zero-pruning fine-tuned control helps separate fine-tuning effects. Hardware Joules are unavailable; process CPU time is an activity proxy, not evidence of energy savings.
