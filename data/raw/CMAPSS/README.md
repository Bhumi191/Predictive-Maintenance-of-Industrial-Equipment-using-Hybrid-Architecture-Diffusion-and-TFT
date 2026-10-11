# NASA C-MAPSS Turbofan Engine Dataset

## Overview
The NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset contains simulated turbofan engine run-to-failure data. It is used for studying engine degradation and predicting Remaining Useful Life (RUL). The dataset includes multivariate time-series sensor measurements collected over successive engine operating cycles.

## Dataset Source
- **Platform:** Kaggle
- **Dataset:** NASA Turbofan Jet Engine Data Set
- **Download URL:** https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

## Dataset Contents
The dataset provides training and testing trajectories for simulated turbofan engines under different operating conditions and fault modes.

Common files include:
- `train_FD001.txt` to `train_FD004.txt` — training trajectories.
- `test_FD001.txt` to `test_FD004.txt` — testing trajectories.
- `RUL_FD001.txt` to `RUL_FD004.txt` — reference Remaining Useful Life values for the corresponding test subsets.

The exact files available depend on the downloaded dataset archive.

## Data Characteristics
- Multivariate time-series data.
- Engine unit identifiers and operating cycles.
- Three operational setting variables.
- Twenty-one sensor measurement variables.
- Simulated degradation progressing toward engine failure.

## Purpose in This Project
This dataset will be used for:
- Time-series preprocessing and feature engineering.
- Engine degradation analysis.
- Remaining Useful Life prediction.
- Developing and evaluating temporal forecasting models.

## Download and Setup
1. Open the Kaggle dataset link above.
2. Sign in to Kaggle if required.
3. Download the dataset archive.
4. Extract the files into this directory.
5. Preserve the original dataset files without modifying them.

## Directory Structure
```text
CMAPSS/
├── README.md
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
└── ...
```

## Data Management
Raw dataset files are excluded from Git tracking to avoid unnecessarily large repository history. Download the dataset separately before running the project.

Processed datasets and engineered features should be stored under `data/processed/`.

## Reference
Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* Proceedings of the International Conference on Prognostics and Health Management.

Refer to the original Kaggle dataset page for licensing and attribution details.