"""Interleaved post-training latency checks to reduce run-order confounding."""
import statistics
import time
import numpy as np
import pandas as pd
import torch
from .models import CNN


@torch.inference_mode()
def interleaved_latency(experiment):
    if (experiment.out/'latency_repetitions.csv').exists():
        print('Using recorded interleaved latency checks.', flush=True)
        return
    models = {}
    for row in experiment.rows:
        if row['dense_artifact'] in models:
            continue
        state = torch.load(experiment.out/row['dense_artifact'], map_location='cpu', weights_only=True)
        model = CNN(state['channels'], state['hidden']).eval()
        model.load_state_dict(state['state_dict'])
        models[row['dense_artifact']] = model
    rows = []
    generator = np.random.default_rng(42)
    for repeat in range(1, 4):
        for position, artifact in enumerate(generator.permutation(list(models))):
            model = models[artifact]
            results = {}
            for batch, runs, warmup in [(1,100,20),(128,20,5)]:
                sample = experiment.tx[:batch]
                for _ in range(warmup):
                    model(sample)
                elapsed = []
                cpu_start = time.process_time_ns()
                for _ in range(runs):
                    start = time.perf_counter_ns()
                    model(sample)
                    elapsed.append((time.perf_counter_ns()-start)/1e6)
                cpu_per_image = (time.process_time_ns()-cpu_start)/1e6/runs/batch
                results[f'batch{batch}_median_ms'] = statistics.median(elapsed)
                if batch == 1:
                    results['inference_cpu_ms_per_image'] = cpu_per_image
            rows.append(dict(repeat=repeat, position=position, dense_artifact=artifact, **results))
        print(f'Interleaved latency round {repeat}/3 completed.', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(experiment.out/'latency_repetitions.csv',index=False)
    aggregate = table.groupby('dense_artifact')[['batch1_median_ms','batch128_median_ms','inference_cpu_ms_per_image']].median()
    pd.DataFrame(experiment.rows).to_csv(experiment.out/'initial_measurements.csv',index=False)
    for row in experiment.rows:
        for column in aggregate.columns:
            row[column] = float(aggregate.loc[row['dense_artifact'],column])
        row['latency_protocol'] = 'median of 3 interleaved rounds; each batch1 100 calls, batch128 20 calls'
    pd.DataFrame(experiment.rows).to_csv(experiment.out/'measurements.csv',index=False)
