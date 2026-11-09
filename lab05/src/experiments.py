"""Gradient validation, matched training and PyTorch execution experiments."""
import gc
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from .value import Value

CONFIG = dict(seed=42, samples=100, noise=0.2, test_fraction=0.2,
              hidden=8, epochs=200, learning_rate=1.0, runs=5,
              optimizer='full-batch SGD', dtype='float64', threads=2)


def save_csv(out, name, rows):
    pd.DataFrame(rows).to_csv(out / name, index=False)


def verify_gradients(out):
    cases = [(2., -3., 1.), (-1.5, 0.7, 2.0), (0.2, 0.3, -0.1), (0., 2., 0.)]
    rows = []
    for case, values in enumerate(cases):
        nodes = [Value(v) for v in values]
        a, b, c = nodes
        loss = ((a*b+c)**2).relu().sum()
        loss.backward()
        tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        expected = ((tensor[0]*tensor[1]+tensor[2])**2).relu().sum()
        expected.backward()

        def function(x):
            return max(0.0, (x[0]*x[1]+x[2])**2)

        for index, label in enumerate('abc'):
            plus, minus = list(values), list(values)
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numerical = (function(plus)-function(minus))/(2e-5)
            custom, reference = nodes[index].grad, float(tensor.grad[index])
            rows.append(dict(case=case, variable=label, custom=custom,
                             pytorch=reference, numerical=numerical,
                             torch_error=abs(custom-reference), numerical_error=abs(custom-numerical)))
    # Shared branches must add their gradient contributions.
    x = Value(3)
    shared = x*x
    loss = shared+shared+x
    loss.backward()
    assert x.grad == 13.0
    order = loss.topological_order()
    positions = {node: i for i, node in enumerate(order)}
    assert all(positions[p] < positions[node] for node in order for p in node.parents)
    assert len(order) == len(set(order))
    assert Value(0).relu().data == 0
    for value, derivative in [(-1., 0.), (0., 0.), (1., 1.)]:
        v = Value(value)
        v.relu().backward()
        assert v.grad == derivative
    table = pd.DataFrame(rows)
    assert table.torch_error.max() < 1e-6
    assert table.numerical_error.max() < 1e-6
    save_csv(out, 'gradient_check.csv', rows)
    return table


def initial_weights():
    rng = np.random.default_rng(CONFIG['seed'])
    return [rng.normal(0, .5, (8, 2)), np.zeros(8), rng.normal(0, .5, (1, 8)), np.zeros(1)]


class ScalarMLP:
    def __init__(self, arrays):
        self.w1 = [[Value(v) for v in row] for row in arrays[0]]
        self.b1 = [Value(v) for v in arrays[1]]
        self.w2 = [Value(v) for v in arrays[2][0]]
        self.b2 = Value(arrays[3][0])
        self.parameters = [v for row in self.w1 for v in row]+self.b1+self.w2+[self.b2]

    def __call__(self, row):
        hidden = [(sum(w*x for w, x in zip(weights, row))+bias).sigmoid()
                  for weights, bias in zip(self.w1, self.b1)]
        return (sum(w*x for w, x in zip(self.w2, hidden))+self.b2).sigmoid()

    def probabilities(self, data):
        # Numeric evaluation avoids constructing graphs outside training.
        w1 = np.array([[v.data for v in row] for row in self.w1])
        b1 = np.array([v.data for v in self.b1])
        w2 = np.array([v.data for v in self.w2])
        hidden = 1/(1+np.exp(-(data@w1.T+b1)))
        return 1/(1+np.exp(-(hidden@w2+self.b2.data)))


def torch_model(arrays):
    model = nn.Sequential(nn.Linear(2, 8), nn.Sigmoid(), nn.Linear(8, 1), nn.Sigmoid()).double()
    with torch.no_grad():
        for param, array in zip(model.parameters(), arrays):
            param.copy_(torch.from_numpy(array))
    return model


def scalar_loss(model, x, y):
    terms = []
    for row, label in zip(x, y):
        probability = model(row)
        terms.append(-probability.log() if label == 1 else -(1-probability).log())
    return sum(terms)/len(terms)


def train_comparison(out):
    x, y = make_moons(n_samples=CONFIG['samples'], noise=CONFIG['noise'], random_state=CONFIG['seed'])
    ids = np.arange(len(y))
    train_ids, test_ids = train_test_split(ids, test_size=.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(x[train_ids])
    x_train, x_test = scaler.transform(x[train_ids]), scaler.transform(x[test_ids])
    y_train, y_test = y[train_ids], y[test_ids]
    tx, ty = torch.tensor(x_train), torch.tensor(y_train, dtype=torch.float64).reshape(-1, 1)
    arrays = initial_weights()
    # Check all 33 parameter gradients on the same initial model and full batch.
    sm, tm = ScalarMLP(arrays), torch_model(arrays)
    s_loss = scalar_loss(sm, x_train, y_train)
    t_loss = nn.functional.binary_cross_entropy(tm(tx), ty)
    s_loss.backward()
    t_loss.backward()
    custom = np.array([p.grad for p in sm.parameters])
    reference = np.concatenate([p.grad.numpy().ravel() for p in tm.parameters()])
    gradient_gap = float(np.max(np.abs(custom-reference)))
    assert gradient_gap < 1e-6
    rows, history = [], []
    final_model = None
    for run in range(1, CONFIG['runs']+1):
        models = {}
        for framework in ['scalar', 'pytorch']:
            gc.collect()
            model = ScalarMLP(arrays) if framework == 'scalar' else torch_model(arrays)
            optimizer = None if framework == 'scalar' else torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'])
            start = time.perf_counter()
            for epoch in range(CONFIG['epochs']):
                if framework == 'scalar':
                    for p in model.parameters:
                        p.grad = 0.0
                    loss = scalar_loss(model, x_train, y_train)
                    loss.backward()
                    for p in model.parameters:
                        p.data -= CONFIG['learning_rate']*p.grad
                    value = loss.data
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.binary_cross_entropy(model(tx), ty)
                    loss.backward()
                    optimizer.step()
                    value = loss.item()
                if run == 1:
                    history.append(dict(framework=framework, epoch=epoch+1, loss=value))
            elapsed = time.perf_counter()-start
            if framework == 'scalar':
                prob = model.probabilities(x_test)
                weights = np.array([p.data for p in model.parameters])
            else:
                with torch.no_grad():
                    prob = model(torch.tensor(x_test)).numpy().ravel()
                weights = np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])
                final_model = model.eval()
            models[framework] = weights
            rows.append(dict(run=run, framework=framework, train_seconds=elapsed,
                             last_training_loss=value, test_accuracy=float(np.mean((prob>=.5)==y_test))))
        assert np.max(np.abs(models['scalar']-models['pytorch'])) < 1e-6
        print(f'Training repetition {run}/5 finished.', flush=True)
    table = pd.DataFrame(rows)
    medians = table.groupby('framework', sort=False)[['train_seconds','last_training_loss','test_accuracy']].median().reset_index()
    save_csv(out, 'training_runs.csv', rows)
    save_csv(out, 'training_history.csv', history)
    medians.to_csv(out/'training_medians.csv', index=False)
    config = dict(CONFIG, split_train_ids=train_ids.tolist(), split_test_ids=test_ids.tolist(),
                  initial_parameter_gradient_max_error=gradient_gap,
                  scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                  timing_scope='200 full-batch forward/backward/SGD steps; excludes setup and evaluation; includes loss.item in PyTorch')
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return table, medians, final_model


def time_calls(function, repeats=30, inner=100, warmup=20):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(inner):
            function()
        samples.append((time.perf_counter_ns()-start)/inner/1e6)
    return statistics.median(samples)


def execution_comparison(out, model):
    torch.manual_seed(42)
    model.eval()
    example = torch.randn(128, 2, dtype=torch.float64)
    started = time.perf_counter()
    traced = torch.jit.freeze(torch.jit.trace(model, example).eval())
    trace_ms = (time.perf_counter()-started)*1000
    (out/'traced_graph.txt').write_text(str(traced.inlined_graph), encoding='utf-8')
    trace_rows = []
    with torch.inference_mode():
        for batch in [1, 128]:
            x = example[:batch]
            error = float((model(x)-traced(x)).abs().max())
            assert error < 1e-12
            eager_ms = time_calls(lambda: model(x))
            traced_ms = time_calls(lambda: traced(x))
            saved = eager_ms-traced_ms
            trace_rows.append(dict(batch=batch, eager_ms=eager_ms, traced_ms=traced_ms,
                                   speedup=eager_ms/traced_ms, trace_setup_ms=trace_ms,
                                   break_even_calls=trace_ms/saved if saved>0 else None,
                                   max_output_error=error, repeats=30, calls_per_repeat=100, warmup_calls=20))
        base = torch.randn(512, 512, dtype=torch.float64)
        view = base.t()
        contiguous = view.contiguous()
        weights = torch.randn(512, 128, dtype=torch.float64)
        assert torch.equal(view, contiguous)
        torch.testing.assert_close(view@weights, contiguous@weights, rtol=1e-12, atol=1e-12)
        tensor_rows = []
        for name, tensor in [('noncontiguous_transpose', view), ('contiguous_copy', contiguous)]:
            tensor_rows.append(dict(tensor=name, shape=str(tuple(tensor.shape)), dtype=str(tensor.dtype),
                                    stride=str(tensor.stride()), contiguous=tensor.is_contiguous(),
                                    shares_base_storage=tensor.untyped_storage().data_ptr()==base.untyped_storage().data_ptr(),
                                    matmul_ms=time_calls(lambda t=tensor: t@weights, inner=20),
                                    repeats=30, calls_per_repeat=20, warmup_calls=20))
        copy_ms = time_calls(lambda: view.contiguous(), inner=20)
    save_csv(out, 'execution_timings.csv', trace_rows)
    save_csv(out, 'tensor_layout.csv', tensor_rows)
    (out/'layout_copy_cost.json').write_text(json.dumps({'contiguous_copy_ms':copy_ms},indent=2),encoding='utf-8')
    versions = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    sklearn=sklearn.__version__, cpu=platform.processor(), device='cpu',
                    torch_threads=torch.get_num_threads(), execution_backend='TorchScript trace + freeze')
    (out/'versions.json').write_text(json.dumps(versions, indent=2),encoding='utf-8')
    return pd.DataFrame(trace_rows), pd.DataFrame(tensor_rows), copy_ms
