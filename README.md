# Where and How Well: Reliability-Guided Dual-View Displacement Learning for Micro-Expression Recognition

This release provides the checkpoints, fold-wise predictions, and scripts needed to reproduce the main reported metrics of "Where and How Well: Reliability-Guided Dual-View Displacement Learning for Micro-Expression Recognition".

## Data

The original SMIC-HS, CASME II, and SAMM datasets, as well as our cropped/processed data, are not included in this package. Please obtain the raw datasets from their official sources and follow their licenses or access requirements.

## Package Structure

```text
.
|-- README.md
|-- requirements.txt
|-- release_manifest.csv
|-- report_metrics.py
|-- artifacts/
|   `-- main_loso/
|       |-- confusion_best_*_norm.png
|       |-- casme2__sub*/
|       |-- samm__*/
|       `-- smic__s*/
|-- model/
|   |-- builder.py
|   |-- dvaw_reco_model.py
|   |-- dvaw_flow_predictor.py
|   |-- motion_classifier.py
|   |-- reco_support/
|   |-- reliability/
|   `-- utils/
```

Each fold directory under `artifacts/main_loso/` contains:

- `best.pt`: released checkpoint;
- `best_y_true.npy`: ground-truth labels for that fold;
- `best_y_pred.npy`: predictions for that fold.

`release_manifest.csv` is a no-image sample manifest containing dataset name, subject id, clip id, onset/apex/offset frame indices, and standardized labels.

## Installation

```bash
pip install -r requirements.txt
```

The scripts were checked with Python `3.10.12`, NumPy `2.2.2`, scikit-learn `1.7.2`, and PyTorch `2.6.0+cu124`.

## Reproduce Reported Metrics

Run from this directory:

```bash
python3 report_metrics.py
```

## Expected Results

| Dataset | UF1 | UAR |
| --- | ---: | ---: |
| 3DB-combined | 0.916 | 0.922 |
| SMIC-HS | 0.887 | 0.893 |
| SAMM | 0.901 | 0.901 |
| CASME II | 0.962 | 0.962 |
