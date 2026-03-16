# Calibration and Data Collection

This repository does not assume any user-specific local path. Replace placeholder values such as `<PATH_TO_CALIBRATION_FILE>` with paths that match your machine.

## Prerequisites

- Python environment with the dependencies used in [`calibration.py`](./calibration.py) and [`data_collection.py`](./data_collection.py)
- `dot_tracking` available in your Python path
- Prophesee / Metavision runtime installed if you are using the live event camera pipeline
- Network access to the robot / gripper controller if you are using the UDP synchronization flow

## Conda environment setup

Create and activate a new conda environment:

```bash
conda create -n tofu python=3.9
conda activate tofu
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If you need the CUDA-enabled PyTorch build, install PyTorch from the official index first, then install the remaining dependencies:

```bash
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

If your machine does not use CUDA, replace the PyTorch packages with a CPU build that matches your platform.

## Recommended file layout

Keep the calibration file inside this repository, for example:

```text
TOFU/
  calibration.py
  data_collection.py
  calibration_result.npz
  tactile_dataset_v6/
```

The default behavior in [`data_collection.py`](./data_collection.py) is:

- use `./calibration_result.npz` relative to the script location
- or use the `TACTILE_CALIBRATION_FILE` environment variable if you set it
- or use `--calibration-file <PATH_TO_CALIBRATION_FILE>` explicitly

## 1. Calibration

Run calibration first to generate the dot reference file:

```bash
python calibration.py --output calibration_result.npz
```

Useful options:

- `--input-event-file <PATH_TO_RAW_OR_ERF>`: use a recorded event file instead of a live camera
- `--grid-rows 9 --grid-cols 7`: marker grid shape
- `--grid-margin-x <VALUE> --grid-margin-y <VALUE>`: adjust the virtual reference grid
- `--dt-step <MICROSECONDS>`: event accumulation interval

Keyboard controls during calibration:

- `Enter`: save calibration
- `C`: reset tracker to the initial reference grid
- `Q` or `Esc`: quit

Output:

- `calibration_result.npz`, containing reference points, measured points, offsets, and radii

## 2. Data collection

After calibration is ready, start the collection script:

```bash
python data_collection.py --calibration-file <PATH_TO_CALIBRATION_FILE> --output-dir tactile_dataset_v6
```

If your calibration file is stored as `./calibration_result.npz`, you can simply run:

```bash
python data_collection.py
```

Useful options:

- `--tm-ip <TM_IP>`
- `--wsl-ip <WSL_OR_UDP_SERVER_IP>`
- `--listen-port <PORT>`
- `--fps 1500`
- `--track-interval 1`
- `--contact-thr 50`
- `--headless`

Keyboard flow in the collector:

- `1` to `8`: choose the material/compliance label and start a trial
- `w`: reopen and re-grip before lifting
- `s`: start the slow-release slip phase
- `d`: finish the trial and save the `.npz`
- `q`: quit

Saved output:

- one compressed `.npz` file per trial under `--output-dir`
- filename format: `<timestamp>_<material>_<compliance>.npz`

## Example commands

Using the default local calibration file:

```bash
python calibration.py --output calibration_result.npz
python data_collection.py --output-dir tactile_dataset_v6
```

Using an explicit placeholder path:

```bash
python calibration.py --output <PATH_TO_CALIBRATION_FILE>
python data_collection.py --calibration-file <PATH_TO_CALIBRATION_FILE> --output-dir <PATH_TO_OUTPUT_DIR>
```

Using an environment variable:

```bash
set TACTILE_CALIBRATION_FILE=<PATH_TO_CALIBRATION_FILE>
python data_collection.py --output-dir tactile_dataset_v6
```
