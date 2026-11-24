# A Local Tool for Organizing Music by Genre

Mariia Liutenko · 73952

This prototype suggests a genre for local music files. Confident suggestions become tags in an exported CSV. Uncertain suggestions wait for a person to choose the final tag. Audio is processed on this computer.

## Start

From the repository root in PowerShell:

```powershell
.\project\start.ps1
```

Open <http://127.0.0.1:8765>. Select one or more MP3, WAV, FLAC or OGG files, click **Suggest genres**, review uncertain suggestions, and save the CSV. The original audio files are unchanged. Stop the server with Ctrl+C.

The launcher uses `project/.venv` if present, otherwise the repository's existing `lab01/.venv`. It does not change the laboratory environment. This checkout has the additional audio reader installed in the ignored `project/.runtime` folder.

For a fresh installation:

```powershell
python -m venv project/.venv
.\project\.venv\Scripts\python.exe -m pip install -r project/requirements.txt
.\project\start.ps1
```

## First version

The pilot uses **560 real FMA-small tracks**: 400 training, 80 validation and 80 test. Each genre has 50/10/10 tracks. These are sampled with seed 42 inside the official splits. This is a smaller experiment than the proposed full 8,000-track study.

Each file becomes a five-second mono segment from its middle, resampled to 22,050 Hz, then a 64-band log-mel spectrogram. A small CNN predicts one of eight genres. Normalization is fitted on training features only and saved with the model.

We choose the checkpoint by validation macro-F1. We choose the confidence threshold using validation labels only. The final test set checks both genre classification and the automatic-tagging decision. If no threshold gives at least 80% validation accuracy with at least five accepted examples, automatic tagging is disabled. The project's full target remains **80% accepted accuracy at 50% coverage**, measured on test; a smaller coverage does not meet that target.

See `results/metrics.json` for measured results and whether each target was reached. The app is a rough prototype, not evidence that all proposal targets have been achieved. Low-confidence and out-of-dataset music still need human checking; softmax confidence is not a calibrated probability of correctness.

The first pilot obtained **42.5% test accuracy** and **0.384 macro-F1**. No validation threshold qualified, so automatic tagging is disabled and all suggestions require review. The model file is **0.129 MB**; warmed CPU forward p95 is **2.60 ms**, and complete processing p95 is **23.07 ms** on this machine. Size and forward-time limits are met; the automatic-tagging quality/coverage target is not met.

## Files

| File | Contents |
|---|---|
| `baseline.ipynb` | Data, one spectrogram, model and measured results |
| `src/download.py` | Reproducible subset download from the official archives; ZIP CRC checks |
| `src/audio.py` | Decode, crop, resample and log-mel conversion |
| `src/model.py` | CNN architecture |
| `src/train.py` | Training, validation threshold, test metrics and latency measurement |
| `src/predict.py` | Saved-model prediction for a local file |
| `src/app.py`, `src/index.html` | Local interface and CSV export |
| `results/manifest.csv` | Track IDs, genres and original splits for this pilot |
| `results/genre_cnn.pt` | Trained TorchScript model |
| `results/model_config.json` | Genre order, selected threshold and run settings |
| `results/metrics.json` | Accuracy, macro-F1, coverage, accepted accuracy, size and latency |
| `results/history.csv` | Training loss and validation metrics by epoch |
| `results/test_predictions.csv` | Test predictions and automatic/review decisions |
| `results/confusion_matrix.csv` | Actual genres in rows, predicted genres in columns |
| `results/timings.csv` | Audio loading, spectrogram, forward and total times per test file |

## Reproduce

Use the Python environment containing the project requirements. Run these commands from the repository root:

```powershell
python -m project.src.download
python -m project.src.train --epochs 20
python -m project.src.predict path/to/song.mp3
```

The first command downloads only selected archive members, rather than the whole 7.2 GiB audio ZIP. Downloaded audio and feature caches stay in ignored `project/data/`. Training uses CPU, two PyTorch threads, Adam, cross-entropy and seed 42. Twenty fixed epochs are used; the best validation checkpoint is retained. Reproduction may vary slightly across software and hardware.

Prediction latency is measured after 20 warm-up forwards, over 200 batch-one forwards from a prepared spectrogram. Audio loading, resampling and spectrogram preparation are timed separately, as well as together with inference. A short file is zero-padded to five seconds. The model only covers the eight FMA-small genres.

The next study is full FMA-small training and then INT8 quantization, with the same held-out split. Neither full-data performance nor INT8 gains are claimed by this pilot.

Data source: [official FMA repository](https://github.com/mdeff/fma).
