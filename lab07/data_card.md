# Fashion-MNIST data card

## Source and license

Fashion-MNIST, Zalando Research (Xiao, Rasul and Vollgraf, 2017): https://github.com/zalandoresearch/fashion-mnist. The source repository provides the data under the MIT license.

## Data and split

70,000 grayscale 28x28 clothing/product images with ten classes. The official 60,000-image training set is split into 55,000 train and 5,000 validation images with a seeded permutation (42). The official 10,000-image test set remains separate. All teacher, pruned and student models use the same records and split; no images are removed or subsampled.

Images are converted to float32, divided by 255 and normalized with mean 0.2860 and standard deviation 0.3530, matching the course preprocessing. Data are prepared in memory; batch shuffling uses a separately seeded generator. No augmentation is applied.

## Quality and limitations

The dataset contains low-resolution catalogue images with visually similar classes. It does not establish performance on arbitrary photographs, backgrounds or real-world camera conditions. Labels and official records are used as supplied. Validation monitors training; checkpoints are not selected by test performance. Test comparisons are descriptive single-seed experiments.

## Storage and privacy

Raw data are cached under repository-level `data/FashionMNIST/` and are excluded from Git. The experiment adds no personal identifiers and performs no linkage to people. Results contain model measurements and weights, not a copy of the image dataset.
