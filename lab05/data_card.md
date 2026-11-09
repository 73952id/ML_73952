# make_moons data card

## Source and purpose

Synthetic two-dimensional binary classification data generated locally with `sklearn.datasets.make_moons`, from the scikit-learn dependency pinned in the repository. No external dataset is downloaded. The experiment compares a scalar automatic-differentiation engine with PyTorch on the same training task.

## Generation and split

100 samples, noise 0.2 and random_state 42. Each sample has two real-valued coordinates and one binary label. An 80/20 train/test split is stratified by label with seed 42. StandardScaler is fitted on the 80 training examples and applied to both partitions. Split IDs and fitted scaling statistics are saved in `results/config.json`.

## Limitations and privacy

The data are artificial, small and generated from two noisy semicircles. They contain no personal records. Test accuracy on 20 synthetic points is only a demonstration and is not evidence of real-world performance. Both implementations use the same generated records; no fairness or population-level claims are made.
