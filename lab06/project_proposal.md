# Individual Project Proposal

**Title:** Compact Music Genre Classification on CPU  
**Student name:** [Full name]  
**Student ID:** 73952

## Problem

Build a compact model that predicts a musical genre from a short audio excerpt. The application could support automatic music-library organization. It will classify audio into the eight FMA-small categories, rather than identify individual songs. The main research question is whether INT8 quantization can reduce model size and CPU inference time while preserving classification quality.

## Dataset and approach

Use FMA-small: 8,000 30-second music excerpts across eight balanced genres, with labels and split information supplied in the FMA metadata. Convert five-second mono excerpts into fixed-size log-mel spectrograms and train a small CNN with an eight-class output. Use random crops during training and a fixed centre crop for validation and testing. Preserve the supplied train/validation/test split; crops from one recording must never cross partitions. Tune model settings on validation data and reserve test data for the final baseline-versus-optimized comparison.

## Success metrics

The primary quality metric is macro-F1, with accuracy and a confusion matrix reported alongside it. The initial target is test macro-F1 of at least 0.60. After quantization, the target is a macro-F1 decrease of no more than 0.02 compared with the FP32 baseline, and at least a twofold reduction in serialized model size. These are project targets to be evaluated experimentally.

## Deployment constraints

Run locally on a documented CPU with batch size 1 and a fixed thread count. The optimized serialized model must be no larger than **5 MB (5,000,000 bytes)**. Target **p95 model inference latency <=50 ms** after warm-up, measured from a prepared spectrogram to the predicted class. Audio decoding and spectrogram construction are excluded from this limit and will be timed separately; the complete processing latency will also be reported. Audio acquisition time is excluded.

## Planned optimization

Train an FP32 CNN baseline, then apply post-training static INT8 quantization with calibration using training data only. Compare quality, serialized size, median latency and p95 latency on the same CPU and evaluation inputs. If post-training quantization causes excessive quality loss, use quantization-aware training. Record configurations, seeds and results to make the comparison reproducible.

**Dataset source:** Defferrard et al., FMA: A Dataset for Music Analysis (2017), https://github.com/mdeff/fma.
