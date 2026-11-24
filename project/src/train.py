"""Train the pilot baseline, choose threshold on validation, evaluate test once."""
import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .audio import ROOT, prepare
from .model import GenreCNN


def predict(model, x):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(part).softmax(1) for part in x.split(32)]).numpy()


def select_threshold(probs, labels):
    confidence = probs.max(1)
    correct = probs.argmax(1) == labels
    candidates = []
    # Require at least five accepted validation examples, avoiding a threshold
    # justified by just one correct prediction in this small pilot.
    for threshold in sorted(set(confidence.tolist())):
        accepted = confidence >= threshold
        if accepted.sum() >= 5 and correct[accepted].mean() >= .80:
            candidates.append((float(accepted.mean()), float(threshold)))
    return max(candidates)[1] if candidates else 1.01


def selective(probs, labels, threshold):
    accepted = probs.max(1) >= threshold
    count = int(accepted.sum())
    accuracy = float((probs.argmax(1)[accepted] == labels[accepted]).mean()) if count else None
    coverage = float(accepted.mean())
    return {'accepted': count, 'total': len(labels), 'coverage': coverage,
            'accuracy_among_accepted': accuracy,
            'target_met': bool(accuracy is not None and accuracy >= .80 and coverage >= .50)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    manifest = pd.read_csv(results / 'manifest.csv')
    assert manifest.track_id.is_unique, 'Repeated track ID in the manifest'
    missing = [row.path for row in manifest.itertuples() if not (ROOT / row.path).exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} audio files missing. Run the downloader first.')
    genres = sorted(manifest.genre.unique().tolist())
    genre_ids = {genre: index for index, genre in enumerate(genres)}
    cache = ROOT / 'data' / 'features.pt'
    ids = manifest.track_id.tolist()
    features = None
    if cache.exists():
        saved = torch.load(cache, map_location='cpu', weights_only=True)
        if saved['track_ids'] == ids:
            features = saved['features']
    if features is None:
        prepared = []
        for index, row in enumerate(manifest.itertuples(), 1):
            feature, _ = prepare(ROOT / row.path)
            prepared.append(feature)
            if index % 40 == 0:
                print(f'Spectrograms {index}/{len(manifest)}', flush=True)
        features = torch.stack(prepared)
        torch.save({'track_ids': ids, 'features': features}, cache)
    labels = torch.tensor([genre_ids[g] for g in manifest.genre], dtype=torch.long)
    parts = {}
    for split in ['training', 'validation', 'test']:
        mask = torch.from_numpy((manifest.split == split).to_numpy())
        parts[split] = (features[mask], labels[mask])
    train_x, train_y = parts['training']
    val_x, val_y = parts['validation']
    test_x, test_y = parts['test']
    model = GenreCNN(train_x.mean().item(), train_x.std().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    history = []
    best_f1 = -1
    best_state = None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        val_probs = predict(model, val_x)
        val_f1 = f1_score(val_y.numpy(), val_probs.argmax(1), average='macro', zero_division=0)
        row = {'epoch': epoch, 'train_loss': total / len(train_y), 'validation_macro_f1': float(val_f1),
               'validation_accuracy': float(accuracy_score(val_y.numpy(), val_probs.argmax(1)))}
        history.append(row)
        print(row, flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    val_probs = predict(model, val_x)
    threshold = select_threshold(val_probs, val_y.numpy())
    # Neither checkpoint selection nor threshold fitting sees the test labels.
    test_probs = predict(model, test_x)
    artifact = results / 'genre_cnn.pt'
    torch.jit.script(model).save(str(artifact))
    deployed = torch.jit.load(str(artifact)).eval()
    sample = test_x[:1]
    timings = []
    with torch.inference_mode():
        for _ in range(20):
            deployed(sample)
        for _ in range(200):
            tick = time.perf_counter()
            deployed(sample)
            timings.append((time.perf_counter() - tick) * 1000)
    assert np.allclose(predict(deployed, test_x), test_probs, atol=1e-6)
    metadata = {'genres': genres, 'threshold': threshold, 'seed': 42,
                'training_scope': 'FMA-small pilot: 50 train, 10 validation, 10 test per genre',
                'segment_seconds': 5, 'segment_position': 'center', 'threads': 2,
                'best_epoch': history[int(np.argmax([h['validation_macro_f1'] for h in history]))]['epoch']}
    (results / 'model_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    test_labels = test_y.numpy()
    metrics = {'scope': metadata['training_scope'], 'seed': 42,
               'split_sizes': {s: len(y) for s, (_, y) in parts.items()},
               'accuracy': float(accuracy_score(test_labels, test_probs.argmax(1))),
               'macro_f1': float(f1_score(test_labels, test_probs.argmax(1), average='macro', zero_division=0)),
               'threshold': threshold, 'validation_selective': selective(val_probs, val_y.numpy(), threshold),
               'test_selective': selective(test_probs, test_labels, threshold),
               'model_bytes': artifact.stat().st_size, 'model_size_limit_met': artifact.stat().st_size <= 5_000_000,
               'forward_p50_ms': float(np.percentile(timings, 50)),
               'forward_p95_ms': float(np.percentile(timings, 95)),
               'latency_limit_met': bool(np.percentile(timings, 95) <= 50),
               'latency_protocol': {'device': 'CPU', 'batch_size': 1, 'threads': 2, 'warmup': 20, 'repeats': 200,
                                    'input': list(sample.shape), 'includes_preprocessing': False},
               'training_seconds': train_seconds, 'python': platform.python_version(), 'torch': torch.__version__,
               'processor': platform.processor(), 'platform': platform.platform()}
    # End-to-end measurements include decode, resampling, mel conversion and inference.
    e2e = []
    for row in manifest.loc[manifest.split == 'test'].itertuples():
        tick = time.perf_counter()
        feature, timing = prepare(ROOT / row.path)
        forward_start = time.perf_counter()
        with torch.inference_mode():
            deployed(feature.unsqueeze(0))
        timing.update(track_id=row.track_id, forward_ms=1000 * (time.perf_counter() - forward_start),
                      total_ms=1000 * (time.perf_counter() - tick))
        e2e.append(timing)
    pd.DataFrame(e2e).to_csv(results / 'timings.csv', index=False)
    metrics['end_to_end_p95_ms'] = float(np.percentile([r['total_ms'] for r in e2e], 95))
    (results / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    pd.DataFrame(history).to_csv(results / 'history.csv', index=False)
    predictions = manifest.loc[manifest.split == 'test'].copy()
    predictions['predicted_genre'] = [genres[i] for i in test_probs.argmax(1)]
    predictions['confidence'] = test_probs.max(1)
    predictions['status'] = np.where(test_probs.max(1) >= threshold, 'auto_tag', 'needs_review')
    predictions.to_csv(results / 'test_predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(test_labels, test_probs.argmax(1), labels=range(len(genres))),
                 index=genres, columns=genres).to_csv(results / 'confusion_matrix.csv')
    report = classification_report(test_labels, test_probs.argmax(1), labels=range(len(genres)),
                                   target_names=genres, output_dict=True, zero_division=0)
    (results / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
