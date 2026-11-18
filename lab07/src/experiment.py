"""Full Fashion-MNIST pruning and distillation experiments for Lab 07."""
import copy
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torchvision.datasets import FashionMNIST

from .models import CNN, magnitude_masks, apply_masks, remove_filters, weight_parameters

CONFIG = dict(seed=42, batch_size=512, teacher_epochs=5, student_epochs=5,
              finetune_epochs=3, learning_rate=.001, finetune_learning_rate=.0001,
              sparsities=[0., .25, .5, .75, .9, .95], filter_fractions=[.1, .25, .5],
              temperature=4., alpha=.5, threads=2, dtype='float32', device='cpu')


class Experiment:
    def __init__(self, root):
        self.root = Path(root)
        self.out = self.root/'results'
        self.out.mkdir(exist_ok=True)
        (self.out/'models').mkdir(exist_ok=True)
        torch.set_num_threads(CONFIG['threads'])
        torch.manual_seed(CONFIG['seed'])
        np.random.seed(CONFIG['seed'])
        raw = FashionMNIST(str(self.root.parent/'data'), train=True, download=True)
        test = FashionMNIST(str(self.root.parent/'data'), train=False, download=True)
        order = torch.randperm(60000, generator=torch.Generator().manual_seed(42))
        train_ids, val_ids = order[:55000], order[55000:]
        def images(data):
            return (data.unsqueeze(1).float()/255-.2860)/.3530
        all_images = images(raw.data)
        self.x, self.y = all_images[train_ids], raw.targets[train_ids]
        self.vx, self.vy = all_images[val_ids], raw.targets[val_ids]
        self.tx, self.ty = images(test.data), test.targets
        self.history, self.rows = [], []
        config = dict(CONFIG, train_rows=len(self.x), validation_rows=len(self.vx), test_rows=len(self.tx),
                      split='seeded permutation of official train; official test unchanged',
                      preprocessing='ToTensor-equivalent /255 then normalize mean .2860 std .3530',
                      pruning_weights='conv/linear weights; biases excluded',
                      latency_rounds=3,
                      runtime_signature=dict(python=platform.python_version(), torch=str(torch.__version__), cpu=platform.processor()),
                      energy='No hardware power meter: report process CPU time as an energy-related proxy, not Joules')
        sources = ''.join((self.root/'src'/name).read_text(encoding='utf-8')
                          for name in ['experiment.py','models.py','latency.py'])
        config['source_sha256'] = hashlib.sha256(sources.encode('utf-8')).hexdigest()
        if (self.out/'measurements.csv').exists():
            previous = json.loads((self.out/'config.json').read_text())
            if previous != config:
                raise ValueError('Recorded results use a different configuration/source; use a separate results directory for a fresh run.')
            self.rows = pd.read_csv(self.out/'measurements.csv').to_dict('records')
            self.history = pd.read_csv(self.out/'training_history.csv').to_dict('records')
        (self.out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        versions = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                        cpu=platform.processor(), device='cpu', threads=torch.get_num_threads())
        if not (self.out/'versions.json').exists():
            (self.out/'versions.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')

    @torch.inference_mode()
    def evaluate(self, model, x, y):
        model.eval()
        correct, loss = 0, 0.
        for start in range(0, len(x), 512):
            logits = model(x[start:start+512])
            labels = y[start:start+512]
            correct += int((logits.argmax(1)==labels).sum())
            loss += float(nn.functional.cross_entropy(logits, labels, reduction='sum'))
        return loss/len(x), correct/len(x)

    def train(self, model, name, epochs, lr, masks=None, teacher=None):
        generator = torch.Generator().manual_seed(42)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        for epoch in range(1, epochs+1):
            model.train()
            if teacher is not None:
                teacher.eval()
            permutation = torch.randperm(len(self.x), generator=generator)
            sum_loss, sum_ce, sum_kl = 0., 0., 0.
            for start in range(0, len(self.x), CONFIG['batch_size']):
                ids = permutation[start:start+CONFIG['batch_size']]
                x, y = self.x[ids], self.y[ids]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                ce = nn.functional.cross_entropy(logits, y)
                kl = logits.new_tensor(0.)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_log_probs = (teacher(x)/CONFIG['temperature']).log_softmax(1)
                    student_log_probs = (logits/CONFIG['temperature']).log_softmax(1)
                    # Exact direction specified by manual: KL(student || teacher).
                    kl = (student_log_probs.exp()*(student_log_probs-teacher_log_probs)).sum(1).mean()
                    loss = CONFIG['alpha']*ce+(1-CONFIG['alpha'])*CONFIG['temperature']**2*kl
                else:
                    loss = ce
                loss.backward()
                if masks is not None:
                    for key, p in model.named_parameters():
                        if key in masks:
                            p.grad.mul_(masks[key])
                optimizer.step()
                if masks is not None:
                    apply_masks(model, masks)
                sum_loss += float(loss.detach())*len(ids)
                sum_ce += float(ce.detach())*len(ids)
                sum_kl += float(kl.detach())*len(ids)
            val_loss, val_accuracy = self.evaluate(model, self.vx, self.vy)
            self.history.append(dict(variant=name, epoch=epoch, train_loss=sum_loss/len(self.x),
                                     train_ce=sum_ce/len(self.x), train_kl=sum_kl/len(self.x),
                                     val_loss=val_loss, val_accuracy=val_accuracy))
            pd.DataFrame(self.history).to_csv(self.out/'training_history.csv', index=False)
            print(f'{name}: epoch {epoch}/{epochs}, loss={sum_loss/len(self.x):.4f}, val={val_accuracy:.2%}', flush=True)
        if masks is not None:
            for name_, p in model.named_parameters():
                if name_ in masks:
                    assert torch.count_nonzero(p.detach()[masks[name_]==0]) == 0
        return dict(training_wall_seconds=time.perf_counter()-start_wall,
                    training_cpu_seconds=time.process_time()-start_cpu)

    def save_formats(self, model, name):
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        dense = self.out/'models'/f'{name}.pt'
        torch.save(dict(channels=model.channels, hidden=model.hidden, state_dict=state), dense)
        arrays = {'__channels':np.array(model.channels), '__hidden':np.array(model.hidden)}
        for key, value in state.items():
            array = value.numpy()
            key = key.replace('.', '__')
            if value.ndim >= 2:
                csr = sp.csr_matrix(array.reshape(array.shape[0], -1))
                arrays[key+'__data'] = csr.data
                arrays[key+'__indices'] = csr.indices.astype(np.int32)
                arrays[key+'__indptr'] = csr.indptr.astype(np.int32)
                arrays[key+'__shape'] = np.array(array.shape, dtype=np.int64)
                restored = sp.csr_matrix((csr.data, csr.indices, csr.indptr), shape=csr.shape).toarray().reshape(array.shape)
                assert np.array_equal(restored, array)
            else:
                arrays[key] = array
        sparse = self.out/'models'/f'{name}_csr.npz'
        # Uncompressed CSR container: savings come from sparsity, not ZIP compression.
        np.savez(sparse, **arrays)
        restored = CNN(model.channels, model.hidden)
        loaded = {}
        with np.load(sparse) as archive:
            for key, value in state.items():
                stem = key.replace('.', '__')
                if value.ndim >= 2:
                    shape = tuple(archive[stem+'__shape'])
                    matrix = sp.csr_matrix((archive[stem+'__data'], archive[stem+'__indices'], archive[stem+'__indptr']),
                                           shape=(shape[0], int(np.prod(shape[1:]))))
                    array = matrix.toarray().reshape(shape)
                else:
                    array = archive[stem]
                loaded[key] = torch.from_numpy(array.copy())
        restored.load_state_dict(loaded)
        for key in state:
            assert torch.equal(restored.state_dict()[key], state[key])
        return dense.stat().st_size, sparse.stat().st_size

    @torch.inference_mode()
    def measure(self, model, name, kind, timing, requested_sparsity=0., filter_fraction=0., extra=None):
        model.eval()
        test_loss, accuracy = self.evaluate(model, self.tx, self.ty)
        sample = self.tx[:1]
        for _ in range(20):
            model(sample)
        batches, batch128 = [], []
        cpu_start = time.process_time_ns()
        for _ in range(100):
            started = time.perf_counter_ns()
            model(sample)
            batches.append((time.perf_counter_ns()-started)/1e6)
        cpu_ms = (time.process_time_ns()-cpu_start)/1e6/100
        for _ in range(5):
            model(self.tx[:128])
        for _ in range(20):
            started = time.perf_counter_ns()
            model(self.tx[:128])
            batch128.append((time.perf_counter_ns()-started)/1e6)
        weights = weight_parameters(model)
        total_weights = sum(p.numel() for _,p in weights)
        nonzero = sum(int(torch.count_nonzero(p)) for _,p in weights)
        dense_bytes, sparse_bytes = self.save_formats(model, name)
        c1, c2 = model.channels
        # Matrix products only, one multiply-add = two FLOPs.
        flops = 2*(9*c1*28*28+9*c1*c2*14*14+c2*49*model.hidden+model.hidden*10)
        nz_ops = 2*sum(int(torch.count_nonzero(p))*({'conv1.weight':28*28,'conv2.weight':14*14}.get(key,1))
                       for key,p in weights)
        row = dict(variant=name, method=kind, requested_weight_sparsity=requested_sparsity,
                   measured_weight_sparsity=1-nonzero/total_weights, requested_filter_fraction=filter_fraction,
                   parameters=sum(p.numel() for p in model.parameters()), nonzero_weights=nonzero,
                   test_accuracy=accuracy, test_loss=test_loss, dense_bytes=dense_bytes, csr_bytes=sparse_bytes,
                   batch1_median_ms=statistics.median(batches), batch128_median_ms=statistics.median(batch128),
                   dense_matmul_flops=flops, ideal_nonzero_flops=nz_ops,
                   inference_cpu_ms_per_image=cpu_ms, data_train_rows=len(self.x), data_test_rows=len(self.tx),
                   energy_joules=None, energy_measurement='unavailable; process CPU time proxy only',
                   dense_artifact=f'models/{name}.pt', csr_artifact=f'models/{name}_csr.npz', **timing)
        if extra:
            row.update(extra)
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.out/'measurements.csv', index=False)
        return row

    def baseline(self):
        torch.manual_seed(42)
        self.teacher = CNN()
        if any(r['variant']=='teacher' for r in self.rows):
            checkpoint = torch.load(self.out/'models'/'teacher.pt', map_location='cpu', weights_only=True)
            self.teacher.load_state_dict(checkpoint['state_dict'])
            print('Loaded recorded teacher checkpoint and measurements.', flush=True)
        else:
            timing = self.train(self.teacher, 'teacher', CONFIG['teacher_epochs'], CONFIG['learning_rate'])
            self.measure(self.teacher, 'teacher', 'baseline', timing)
        self.teacher.requires_grad_(False).eval()
        return pd.DataFrame(self.rows).query("variant == 'teacher'")

    def pruning(self):
        control = None
        for scope in ['global', 'layerwise']:
            for sparsity in CONFIG['sparsities']:
                name = f'{scope}_{int(sparsity*100):02d}'
                existing = next((r for r in self.rows if r['variant']==name), None)
                if existing is not None:
                    if sparsity == 0:
                        control = existing
                    print(f'Loaded recorded {name} measurements.', flush=True)
                    continue
                if sparsity == 0 and control is not None:
                    row = dict(control, variant=name, method=scope,
                               reused_control='identical 0% fixed-mask control; not a second independent run')
                    self.rows.append(row)
                    pd.DataFrame(self.rows).to_csv(self.out/'measurements.csv', index=False)
                    continue
                model = copy.deepcopy(self.teacher).requires_grad_(True)
                masks = magnitude_masks(model, sparsity, scope)
                apply_masks(model, masks)
                timing = self.train(model, name, 3, CONFIG['finetune_learning_rate'], masks=masks)
                row = self.measure(model, name, scope, timing, requested_sparsity=sparsity,
                                   extra=dict(mask_fixed_verified=True))
                if sparsity == 0:
                    control = row
        self.structured = []
        for fraction in CONFIG['filter_fractions']:
            torch.manual_seed(42)
            model, details = remove_filters(self.teacher, fraction)
            name = f'structured_{int(fraction*100):02d}'
            existing = next((r for r in self.rows if r['variant']==name), None)
            if existing is not None:
                self.structured.append(existing)
                print(f'Loaded recorded {name} measurements.', flush=True)
                continue
            timing = self.train(model, name, 3, CONFIG['finetune_learning_rate'])
            self.structured.append(self.measure(model, name, 'structured', timing,
                                               filter_fraction=fraction, extra=details))
        table = pd.DataFrame(self.rows)
        fig, axes = plt.subplots(1, 2, figsize=(11,4))
        for kind in ['global','layerwise']:
            data = table[table.method==kind]
            axes[0].plot(100*data.measured_weight_sparsity,100*data.test_accuracy,'o-',label=kind)
        axes[0].set(xlabel='Weight sparsity (%)',ylabel='Test accuracy (%)', title='Fixed-mask magnitude pruning; 3 FT epochs')
        axes[0].legend()
        data = table[table.method=='structured']
        baseline_accuracy = table.loc[table.variant=='teacher','test_accuracy'].iloc[0]
        axes[1].plot([0]+[100*v for v in data.actual_filter_fraction_removed],
                     [100*baseline_accuracy]+[100*v for v in data.test_accuracy],'o-')
        axes[1].set(xlabel='Actual filters removed (%)',ylabel='Test accuracy (%)',title='Physical L2 filter removal; 3 FT epochs')
        for ax in axes:
            ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(self.out/'sparsity_curves.png',dpi=160)
        plt.close(fig)
        return table[~table.method.isin(['plain_student','distilled'])]

    def storage_latency(self):
        table = pd.DataFrame(self.rows)
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        for kind in ['global','layerwise']:
            data = table[table.method==kind]
            axes[0].plot(100*data.measured_weight_sparsity, data.batch1_median_ms,'o-',label=kind)
        axes[0].set(xlabel='Weight sparsity (%)',ylabel='Dense CPU batch-1 median (ms)')
        data = table[table.method=='global']
        axes[1].plot(100*data.measured_weight_sparsity,data.dense_bytes/1e6,'o-',label='Dense state')
        axes[1].plot(100*data.measured_weight_sparsity,data.csr_bytes/1e6,'o-',label='Uncompressed CSR container')
        axes[1].set(xlabel='Weight sparsity (%)',ylabel='File size (MB, decimal)')
        for ax in axes:
            ax.grid(alpha=.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.out/'storage_latency.png',dpi=160)
        plt.close(fig)
        return table.loc[~table.method.isin(['plain_student','distilled']),
                         ['variant','parameters','dense_bytes','csr_bytes','batch1_median_ms','dense_matmul_flops']]

    def distillation(self):
        for distilled in [False,True]:
            torch.manual_seed(42)
            model = CNN(hidden=128)
            name = 'student_distilled' if distilled else 'student_plain'
            if any(r['variant']==name for r in self.rows):
                print(f'Loaded recorded {name} measurements.', flush=True)
                continue
            timing = self.train(model, name, CONFIG['student_epochs'], CONFIG['learning_rate'],
                                teacher=self.teacher if distilled else None)
            self.measure(model, name, 'distilled' if distilled else 'plain_student', timing)
        return pd.DataFrame(self.rows).query("variant in ['teacher','student_plain','student_distilled']")

    def summary(self):
        table = pd.DataFrame(self.rows)
        columns = ['variant','data_train_rows','data_test_rows','test_accuracy','parameters','nonzero_weights',
                   'dense_bytes','csr_bytes','dense_matmul_flops','batch1_median_ms',
                   'training_wall_seconds','training_cpu_seconds','inference_cpu_ms_per_image','energy_joules']
        table[columns].to_csv(self.out/'efficiency_pillars.csv',index=False)
        return table[columns]
