# Industrial IoT Dataset (Synthetic)

## Overview
The Industrial IoT Dataset (Synthetic) is a synthetically generated dataset designed for industrial Internet of Things (IIoT) analytics. It supports research and experimentation involving predictive maintenance, equipment condition monitoring, and anomaly or failure analysis.

Because the data is synthetic, its patterns represent simulated industrial conditions rather than direct measurements from a real industrial deployment.

## Dataset Source
- **Platform:** Kaggle
- **Dataset:** Industrial IoT Dataset (Synthetic)
- **Creator:** Can Özensoy
- **Download URL:** https://www.kaggle.com/datasets/canozensoy/industrial-iot-dataset-synthetic

## Dataset Contents
Download the original dataset files from the Kaggle page and preserve their supplied filenames and formats.

Consult the dataset's description and file documentation for the exact columns, sensor variables, target labels, and data-generation assumptions. Do not assume that particular sensor fields or failure labels are present until verified.

## Purpose in This Project
This dataset will be used to investigate:
- Industrial sensor-data preprocessing.
- Exploratory data analysis and feature relationships.
- Equipment condition and failure-pattern analysis.
- Anomaly detection and predictive-maintenance workflows.
- Evaluation of suitable machine-learning approaches.

## Download and Setup
1. Open the Kaggle dataset link above.
2. Sign in to Kaggle if required.
3. Download the dataset archive.
4. Extract the downloaded files into this directory.
5. Keep the original files unchanged and document any preprocessing separately.

## Directory Structure
```text
Industrial-IoT/
├── README.md
└── [original downloaded dataset files]
```

The actual filenames and subdirectories will depend on the downloaded archive.

## Data Management
Raw dataset files are excluded from Git tracking. Download the dataset separately before running the project.

Store cleaned data, transformed features, and other derived outputs under `data/processed/`.

## Reproducibility and Limitations
- This is a synthetic dataset, not a substitute for validation on real industrial sensor data.
- Review the original documentation before interpreting variables or labels.
- Avoid data leakage when splitting observations into training and testing sets.
- Record preprocessing steps and experimental settings for reproducibility.

## License and Attribution
Refer to the original Kaggle dataset page for the applicable license, attribution requirements, and permitted uses.