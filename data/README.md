# Dataset Directory

This project uses the **WISDM Smartphone and Smartwatch Activity and Biometrics Dataset** (51 subjects, 20 Hz tri-axial accelerometer & gyroscope recordings across 18 daily activities).

## Directory Layout

- `data/raw/`: Optional local mirror for WISDM raw files (`phone/accel/*.txt`, `phone/gyro/*.txt`, `watch/accel/*.txt`, `watch/gyro/*.txt`). By default, `configs/config.yaml` points directly to your existing external `wisdm-dataset/data/raw` folder so large raw files are not duplicated.
- `data/processed/`: Generated NumPy window splits (`X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`) produced by `scripts/prepare_dataset.py`.

## Raw Line Format

Each `.txt` file contains comma-separated rows terminated by a semicolon:

```text
<user_id>,<activity_code>,<timestamp_ns>,<x>,<y>,<z>;
```
