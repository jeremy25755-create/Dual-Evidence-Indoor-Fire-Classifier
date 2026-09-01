# Dual-Evidence Indoor Fire Classifier

A lightweight indoor-fire classifier that models uncertainty separately from
sensor-response levels and temporal trends, then fuses the two evidence streams
to classify **Background**, **Fire**, and **Nuisance**.

The repository also retains the TEFN-based classification baseline used for
comparison. The proposed model is in `Fire_Semantic_Evidence_Lite/`.

## Dataset

`dataset/Indoor Fire Dataset with Distributed Multi-Sensor Nodes.csv` is the
public dataset used in the experiments.

- Pascal Vorwerk, *Indoor Fire Dataset with Distributed Multi-Sensor Nodes*,
  Mendeley Data, Version 1, 2023.
- DOI: https://doi.org/10.17632/npk2zcm85h.1
- License: CC BY 4.0

## Setup

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
```

Use `requirements-cpu.txt` instead of `requirements-gpu.txt` on a CPU-only
machine.

## Train the proposed model

```powershell
& ".\.venv\Scripts\python.exe" ".\Fire_Semantic_Evidence_Lite\train.py" `
  --data ".\dataset\Indoor Fire Dataset with Distributed Multi-Sensor Nodes.csv" `
  --seq-len 60 --stride 10 --embedding-dim 32 `
  --epochs 30 --batch-size 128 --runs 5 --device cuda `
  --output ".\results\dual_evidence_5seeds"
```

The final report includes mean and standard deviation for accuracy, macro F1,
weighted F1, and class-wise F1, recall, and precision.

## Test

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests\test_fire_semantic_evidence_lite.py tests\test_data_provider.py
```

Generated checkpoints and result folders are intentionally excluded from Git.

