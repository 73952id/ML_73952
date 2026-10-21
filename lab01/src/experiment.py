
"""Lab 01: compare two classifiers and measure their resource costs."""

import csv
import gc
import random
import statistics
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def measure_training(factory, X_train, y_train):
    """Measure training time and sampled peak process memory."""
    model = factory()
    start = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)

        peak_mb, _ = memory_usage(
            (model.fit, (X_train, y_train), {}),
            interval=0.001,
            max_usage=True,
            retval=True,
            include_children=False,
            multiprocess=False,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    nonconverged = any(
        issubclass(w.category, ConvergenceWarning)
        for w in caught
    )

    return model, elapsed_ms, peak_mb, nonconverged


def measure_inference(model, sample):
    """Measure 100 predictions and sampled peak memory."""
    times = []

    def predict_100():
        for _ in range(100):
            start = time.perf_counter_ns()
            model.predict(sample)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            times.append(elapsed)

    peak_mb = memory_usage(
        (predict_100, (), {}),
        interval=0.001,
        max_usage=True,
        include_children=False,
        multiprocess=False,
    )

    return statistics.median(times), peak_mb


def write_csv(name, rows):
    """Save experiment results to a CSV file."""
    with (OUT / name).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(42)
    np.random.seed(42)

    OUT.mkdir(parents=True, exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    models = {
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    accuracies = []
    measurements = []

    for name, factory in models.items():
        print(f"Measuring {name}...")

        #Warm-up before the five training runs.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", ConvergenceWarning
            )
            factory().fit(X_train, y_train)

        gc.collect()

        training_times = []
        training_peaks = []
        convergence_warning = False

        #Five measured training runs.
        for _ in range(5):
            gc.collect()

            model, elapsed, peak, warning = measure_training(
                factory, X_train, y_train
            )

            training_times.append(elapsed)
            training_peaks.append(peak)

            convergence_warning |= warning

        #Evaluate the final trained model.
        accuracy = model.score(X_test, y_test)

        #One sample for inference benchmarking.
        sample = X_test[:1]

        #Inference warm-up.
        model.predict(sample)

        inference_median, inference_peak = measure_inference(
            model, sample
        )

        #Save the model and measure file size.
        filename = OUT / f"{name.lower()}_model.joblib"
        joblib.dump(model, filename)

        size_bytes = filename.stat().st_size

        accuracies.append({
            "model": name,
            "test_accuracy": f"{accuracy:.4f}",
        })

        measurements.append({
            "model": name,
            "training_median_ms": (
                f"{statistics.median(training_times):.4f}"
            ),
            "inference_median_ms": (
                f"{inference_median:.6f}"
            ),
            "model_bytes": size_bytes,
            "model_kb": f"{size_bytes / 1024:.3f}",
            "training_sampled_peak_rss_mb": (
                f"{max(training_peaks):.3f}"
            ),
            "inference_sampled_peak_rss_mb": (
                f"{inference_peak:.3f}"
            ),
            "convergence_warning": convergence_warning,
        })

    write_csv("baseline_accuracy.csv", accuracies)
    write_csv("measurements.csv", measurements)

    for row in measurements:
        print(row)

    print(
        "Memory values are sampled peaks of the "
        "Python process (not model-only memory)."
    )


if __name__ == "__main__":
    main()
