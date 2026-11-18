"""Build the Lab 07 report from saved results."""
from pathlib import Path
import pandas as pd


def build_report(root):
    root = Path(root)
    table = pd.read_csv(root/'results'/'measurements.csv')

    def get(name):
        return table.loc[table.variant == name].iloc[0]

    teacher = get('teacher')
    plain = get('student_plain')
    distilled = get('student_distilled')
    sparse = get('global_95')
    control = get('global_00')
    best = table.loc[table.test_accuracy.idxmax()]
    names = {'teacher': 'Teacher', 'student_plain': 'Plain student',
             'student_distilled': 'Distilled student'}

    def label(name):
        if name in names:
            return names[name]
        method, percent = name.rsplit('_', 1)
        return f"{method.capitalize()} {int(percent)}%"

    text = f'''# Lab 07: Pruning and distillation

## 1. Baseline CNN

The aim of this lab was to make a CNN smaller and compare what happens to its accuracy, file size and prediction time.

I used Fashion-MNIST with 55,000 training images, 5,000 validation images and the official 10,000 test images. All models used the same split and normalization. The seed was 42.

The baseline CNN has two convolutional layers with 8 and 16 filters, followed by pooling and a 784-1280-10 classifier. It has **{int(teacher.parameters):,} parameters**, which is close to the required one million. I trained it for five epochs using Adam, a learning rate of 0.001 and batch size 512. The experiments ran on CPU with two PyTorch threads.

The final model reached **{100*teacher.test_accuracy:.2f}% test accuracy**. Its dense file was **{teacher.dense_bytes/1e6:.3f} MB**, and its measured prediction time for one image was **{teacher.batch1_median_ms:.4f} ms**.

## 2. Pruning

First, I removed weights with the smallest absolute values. I tested 0%, 25%, 50%, 75%, 90% and 95% sparsity in two ways:

- **Global:** choose the smallest weights across the whole model.
- **Layer-wise:** remove the same percentage of weights from each layer.

Biases were kept. Every variant started from the original baseline and was fine-tuned for three epochs with Adam at learning rate 0.0001. A fixed mask kept the removed weights at zero throughout training.

I also fine-tuned a model with no weights removed. It reached **{100*control.test_accuracy:.2f}%**, compared with the original **{100*teacher.test_accuracy:.2f}%**. This matters because some improvement can come from the extra training itself. The global and layer-wise 0% cases are identical, so they share this control run.

Next, I tried removing whole convolutional filters. I ranked filters by their L2 norm and removed those with the smallest norms. I also adjusted the following layers so that the resulting network was physically smaller. The requested levels were 10%, 25% and 50%; rounding to whole filters gave 12.5%, 25% and 50%. Each model was then fine-tuned for three epochs.

![Accuracy after pruning](results/sparsity_curves.png)

Global pruning kept accuracy fairly close to the control through 75% sparsity. At higher sparsity, layer-wise pruning lost more accuracy because it also removed a large share of the convolutional weights. In the structured plot, the 0% point is the original baseline; the other points include fine-tuning.

## 3. File size and prediction time

I saved each model in a normal dense format and in an uncompressed CSR format. CSR stores nonzero weights and their positions. I checked that loading the CSR arrays restored exactly the same weights.

For the baseline, the dense file was **{teacher.dense_bytes/1e6:.3f} MB**, while CSR took **{teacher.csr_bytes/1e6:.3f} MB**. CSR was larger because storing positions adds overhead. At 95% global sparsity, CSR fell to **{sparse.csr_bytes/1e6:.3f} MB**, while the dense file stayed at **{sparse.dense_bytes/1e6:.3f} MB**.

A smaller sparse file does not automatically make prediction faster. The pruned models still ran through normal dense PyTorch operations, which process the full tensors, including zeros. Removing whole filters really changes tensor sizes and reduces the required calculations.

I measured prediction time after warm-up, without data loading or file conversion. Each round used 100 calls for batch 1 and 20 for batch 128, after 20 and 5 warm-up calls respectively. I repeated this in three rounds with shuffled model order and used the median of the round medians.

![File size and prediction time](results/storage_latency.png)

Timings varied noticeably between rounds. Even the two students, which have the same architecture, had different measured times. These results should therefore be read as measurements from this computer, rather than proof that one weight pattern is faster. All rounds are saved in `latency_repetitions.csv`.

## 4. Knowledge distillation

The student uses the same convolutional layers but reduces the hidden classifier from 1280 units to 128. This gives **{int(plain.parameters):,} parameters**. I trained it once using only the correct labels and once using both the labels and the teacher's predictions.

Both students started with the same weights and used the same batches, Adam settings and five training epochs. The teacher was frozen. Following the manual, I used T=4, alpha=0.5 and this loss:

`0.5 * CE(student, labels) + 0.5 * T^2 * KL(student probabilities || teacher probabilities)`

The probabilities in the KL term use logits divided by T. The KL direction is student-to-teacher, as written in the manual.

The plain student reached **{100*plain.test_accuracy:.2f}%**, and the distilled student reached **{100*distilled.test_accuracy:.2f}%**. Distillation improved accuracy by **{100*(distilled.test_accuracy-plain.test_accuracy):.2f} percentage points** in this run. The improvement was small, and training took extra work because it also needed the teacher's forward pass.

## 5. Overall comparison

The table brings together the four efficiency pillars:

- **Data:** every model used the same 55,000 training and 10,000 test images, plus 5,000 validation images.
- **Model:** parameter count, nonzero weights and saved file sizes.
- **Compute:** FLOPs and measured prediction time.
- **Energy:** CPU time as a rough activity indicator. I could not measure energy in Joules. CPU time is not an energy reading and cannot prove energy savings.

MB means 1,000,000 bytes. FLOPs count multiplication and addition separately and exclude biases, activations and pooling. Zero weights do not reduce the dense FLOP count. CPU time includes work across threads and can be higher than elapsed time.

| Model | Train/test images | Accuracy % | Parameters | Nonzero weights | Dense MB | CSR MB | MFLOPs | Time/image ms | CPU ms/image* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for _, row in table.iterrows():
        text += (f'| {label(row.variant)} | 55000/10000 | {100*row.test_accuracy:.2f} '
                 f'| {int(row.parameters):,} | {int(row.nonzero_weights):,} '
                 f'| {row.dense_bytes/1e6:.3f} | {row.csr_bytes/1e6:.3f} '
                 f'| {row.dense_matmul_flops/1e6:.3f} | {row.batch1_median_ms:.4f} '
                 f'| {row.inference_cpu_ms_per_image:.4f} |\n')
    text += f'''
*CPU time is only an activity indicator, not measured energy. The two 0% pruning rows share one control run. Training times in the result files include validation; pruning times cover the three extra epochs only.

### Conclusion

The highest accuracy was **{100*best.test_accuracy:.2f}%**, from **{label(best.variant)}**. Global pruning preserved accuracy better than layer-wise pruning at high sparsity. Sparse storage helped once enough weights were zero, but it did not reduce the work done by dense kernels.

The student used about **{teacher.parameters/plain.parameters:.1f} times fewer parameters** than the teacher. Distillation gave it a small accuracy improvement of **{100*(distilled.test_accuracy-plain.test_accuracy):.2f} percentage points**. Removing filters and using a smaller network reduced the actual dense computation, while prediction times still needed careful measurement.

These are single-seed results. Timing variation was large enough that small speed differences are not reliable. Energy savings cannot be confirmed without power measurements.

## Sources

Lab Manual: Labs 07, Compression: pruning and distillation; Reddi, Chapters 9 and 10, sections 10.3-10.4. Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist. Data and model details are in `data_card.md` and `model_card.md`.
'''
    (root/'report.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    build_report(Path(__file__).resolve().parents[1])
