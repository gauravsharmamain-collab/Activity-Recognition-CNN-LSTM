# Activity Recognition CNN-LSTM

Human Activity Recognition (HAR) from multi-axis smartphone and smartwatch inertial sensor streams using a hybrid **1D-CNN + stacked LSTM** architecture in PyTorch, configured for the **WISDM Smartphone and Smartwatch Activity and Biometrics Dataset**.

## Architecture

```
Input (B, 100, 3) ──▶ Conv1D(3 → 64, k=3) + ReLU + Dropout(0.5)
                 ──▶ Stacked LSTM(64 → 100, 2 layers, dropout=0.5)
                 ──▶ Last Timestep ──▶ Dropout(0.5) ──▶ Linear(100 → 100) + ReLU ──▶ Linear(100 → 6)
```

### Target Activity Classes

| Class ID | Label | WISDM Activity Code(s) |
|---|---|---|
| `0` | Walking | `A` |
| `1` | Jogging | `B` |
| `2` | Stairs | `C` |
| `3` | Sitting | `D` |
| `4` | Standing | `E` |
| `5` | Eating | `H` (soup), `I` (chips), `J` (pasta), `L` (sandwich) |

## Repository Structure

```text
Activity-Recognition-CNN-LSTM/
├── configs/config.yaml        # Dataset paths, windowing, model & training config
├── data/                      # Raw/processed placeholders & dataset guide
├── notebooks/                 # 5-stage interactive Jupyter workflow
├── src/
│   ├── data/                  # WISDM parsing, cleaning, scaling, segmentation
│   ├── features/              # Time-domain features & EDA plots
│   ├── models/                # PyTorch CNNLSTMClassifier
│   ├── training/              # Training loop, checkpointing, learning curves
│   ├── evaluation/            # Confusion matrix, F1, ROC-AUC, reports
│   └── inference/             # ActivityPredictor for windows & DataFrames
├── scripts/                   # End-to-end CLI entrypoints
├── tests/                     # Unit tests (preprocessing, segmentation, model)
├── models/                    # Saved checkpoints & fitted scaler stats
└── results/                   # Plots, JSON metrics, and text reports
```

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify dataset path** in [`configs/config.yaml`](configs/config.yaml) (`data.raw_dir` already points to your local WISDM `data/raw` folder, with `data/raw` as fallback).

3. **Prepare segmented train/val/test splits**

   ```bash
   python scripts/prepare_dataset.py --config configs/config.yaml
   ```

4. **Train the CNN-LSTM model**

   ```bash
   python scripts/train_model.py --config configs/config.yaml
   ```

5. **Evaluate on the test split**

   ```bash
   python scripts/evaluate_model.py --config configs/config.yaml
   ```

6. **Run unit tests**

   ```bash
   pytest tests
   # or without pytest:
   python -m unittest discover -s tests
   ```
