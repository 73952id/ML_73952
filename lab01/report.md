# Lab 01 - Environment and first system measurements

## 1. Goal

The goal of this laboratory was to train and compare two machine learning models, Logistic Regression and Random Forest, using the Breast Cancer Wisconsin dataset.

The experiment focused on prediction accuracy, training time, inference latency, model size, and memory consumption. The results were used to evaluate the models' suitability for different deployment environments.

## 2. Method

The experiment was conducted using Python 3.11.8 and scikit-learn.

The Breast Cancer Wisconsin dataset contains 569 samples and 30 numerical features. The dataset was split into 70% training data and 30% testing data, using stratification and random_state=42.

Two models were trained:

- Logistic Regression with max_iter=1000 and random_state=42.
- Random Forest with 100 estimators and random_state=42.

Training time was measured five times after one warm-up run, and the median was reported. Single-sample inference was measured 100 times after a warm-up.

Model sizes were measured using joblib serialization. Memory was monitored using memory_profiler, with RSS sampled every 1 ms during training and inference. These are sampled peak values for the Python process, not memory used by the model alone.

## 3. Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Test accuracy | 0.9415 | 0.9357 |
| Training time (ms) | 1,436.6008 | 545.8156 |
| Inference latency (ms) | 0.171500 | 10.030050 |
| Model size (bytes) | 1,055 | 290,889 |
| Model size (KB) | 1.030 | 284.071 |
| Training sampled peak RSS (MB) | 229.758 | 230.262 |
| Inference sampled peak RSS (MB) | 229.699 | 230.336 |

Logistic Regression produced a convergence warning because the optimizer reached the maximum number of iterations.

The memory values are sampled peak RSS measurements. Short spikes between samples may not have been captured. Training time measures fit directly, separately from an additional profiled fit per repetition. The memory profiler is restricted to one execution per measurement. Torch is imported to set the required seeds, so its runtime contributes to process RSS. Timings and memory were regenerated after restructuring the project.

## 4. Deployment Suitability

| Target | Available memory | Latency limit | Model size limit |
|---|---|---|---|
| Cloud | >= 1 GB | <= 100 ms | <= 500 MB |
| Edge | 256-1024 MB | <= 50 ms | <= 50 MB |
| Mobile | 64-256 MB | <= 20 ms | <= 10 MB |
| TinyML | <= 256 KB | <= 10 ms | <= 100 KB |

Available memory is a device capacity, not a minimum amount the process must consume. KB and MB in the measurement files use powers of 1024.

**Cloud:** Both models satisfy the measured latency and model-size limits. Their sampled process memory usage is small relative to a typical cloud environment.

**Edge:** Both models meet the specified latency and model-size budgets, with sampled process RSS below even the 256 MB lower-capacity edge device.

**Mobile:** Both models satisfy the specified latency and model-size budgets. Their observed process memory fits a 256 MB device but exceeds a 64 MB device, although actual mobile deployment would require additional testing.

**TinyML:** Neither model can be confirmed suitable in its current Python implementation. Logistic Regression has a sufficiently small serialized model, but its sampled process memory exceeds the TinyML budget. Random Forest additionally exceeds the 100 KB model-size limit.

These conclusions are based on measurements from the development computer, not direct benchmarks on target devices.

## 5. Conclusions

1. I expected Random Forest to be more accurate, but Logistic Regression did slightly better on my test data (94.15% versus 93.57%). The difference is small, so I would not say it is always the better model.

2. Logistic Regression made predictions much faster and its saved file was much smaller. However, it took longer to train in the repeated measurements and the optimizer reached its iteration limit.

3. I learned that accuracy is only one part of choosing a model. Even when a model predicts well, its memory use, file size and speed can make it unsuitable for some devices.

## 6. Limitations

The Logistic Regression model did not fully converge under the specified iteration limit.

Memory usage was estimated through sampled process RSS, which may miss short-lived memory peaks and includes Python interpreter and library overhead.

The deployment assessment is preliminary because the models were not tested directly on Cloud, Edge, Mobile, or TinyML hardware.
## 7. Environment Note

The lab manual requests `pyarrow==16.1.0`, but this conflicts with `mlflow==2.14.1`, which requires `pyarrow<16`. I used `pyarrow==15.0.2` for installation. This exception needs instructor approval.
