# In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis.

Code used for In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis.
(Part of our code is adopted from https://github.com/helme/ecg_ptbxl_benchmarking.)

## Setup

`pip install poetry`
(Use python 3.8 or later)

## Data preparation

```
cd preparation
./get_dataset.sh
```
For G12EC dataset, manually download from [link](https://www.kaggle.com/bjoernjostein/georgia-12lead-ecg-challenge-database/metadata).

## Experiment

```
cd experiment
```

For grid search: `poetry run python experiment_0.py 0`

For multi-label classification: `poetry run python experiment_1.py 0`

For multi-class classification: `poetry run python experiment_2.py 0`
