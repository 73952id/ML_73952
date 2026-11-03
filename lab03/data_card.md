# Fashion-MNIST data card

## Source and license

Fashion-MNIST by Han Xiao, Kashif Rasul and Roland Vollgraf, Zalando Research (2017). Source and documentation: https://github.com/zalandoresearch/fashion-mnist. The source repository distributes the dataset under the MIT license: https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE.

## Data and task

70,000 grayscale product images, each 28 x 28 pixels, with ten labels: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. The task is single-label image classification.

The official split has 60,000 training and 10,000 test examples. Lab 03 splits official training into 55,000 train and 5,000 validation images with generator seed 42; train and validation indices are checked to be disjoint. Official test data remain separate.

## Processing

Loaded with `torchvision.datasets.FashionMNIST`. ToTensor converts image values to 0-1; Normalize uses the manual's fixed mean 0.2860 and standard deviation 0.3530. Batch size is 128. Training batches are shuffled with a seeded generator; validation and test batches are not shuffled. All three models use the same split and preprocessing.

## Limitations and privacy

Small grayscale catalogue images do not represent the full variation of real-world clothing photos, backgrounds, lighting or current products. Visually overlapping clothing classes can be difficult to distinguish at 28 x 28 resolution. Generalization to other image sources and demographic fairness are not evaluated.

The experiment uses product images and category labels without added personal identifiers. No linkage to customer records is performed.

## Storage and use

Raw files are cached in the ignored repository-level `data/FashionMNIST/` directory. They are not copied into the lab results. Intended use is educational architecture and computational-cost comparison.
