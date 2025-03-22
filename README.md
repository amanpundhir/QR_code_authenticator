# QR Code Authentication: Detecting Original vs. Counterfeit Prints

This repository contains code and documentation for the QR Code Authentication project. The goal is to develop a system that distinguishes between authentic (first print) and counterfeit (second print) QR codes using two approaches:
1. **Traditional computer vision** (LBP, HOG + Random Forest).
2. **Deep learning** (custom CNN and MobileNetV2 transfer learning).

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments and Results](#experiments-and-results)
- [Report](#report)

## Overview
QR codes are vulnerable to counterfeiting due to easy replication. This project compares original prints with counterfeit versions using:
- **Traditional Approach**: Combines LBP/HOG features with a Random Forest classifier.
- **Deep Learning Approach**: Uses a custom CNN and MobileNetV2 for feature extraction and classification.


## Project Structure
```plaintext
QR_Code_Authentication_Project/
│
├── README.md
├── requirements.txt
├── data/
│   ├── first_print/First Print/     # Original QR codes
│   └── second_print/Second Print/   # Counterfeit QR codes
├── src/
│   ├── data_preparation.py          # Dataset organization
│   ├── feature_engg.py              # LBP/HOG extraction
│   ├── traditional_model.py         # Random Forest pipeline
│   ├── cnn_model.py                 # Custom CNN
│   ├── transfer learning_model.py   # MobileNetV2 transfer learning
│   └── evaluation.py                # Metrics and plots
├── notebooks/                       # Experimental notebooks
├── models/                          # Saved models
│   ├── transfer_learning.tflite
│   └── traditional_model.joblib
└── report/                          # Final report
    └── QR_Code_Authentication_Report.pdf
```

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/amanpundhir/QR_code_authenticator.git
   cd QR_code_authenticator
   ```

2. **Install dependencies** (Python 3.7+ required):
   ```bash
   pip install -r requirements.txt
   ```
   - Packages: `opencv-python`, `matplotlib`, `scikit-image`, `scikit-learn`, `tensorflow`, `joblib`, `gdown`.

## Usage
### Data Preparation
- Place raw dataset files in `data/`.
- Organize data for CNN training:
  ```bash
  python src/data_preparation.py  # or use the provided Colab notebook
  ```

### Training Models
- **Traditional CV Pipeline**:
  ```bash
  python src/traditional_model.py
  ```
- **Custom CNN**:
  ```bash
  python src/cnn_model.py
  ```
- **MobileNetV2 Transfer Learning**:
  ```bash
  python src/transfer learning_model.py
  ```

### Evaluation
Generate performance metrics and plots:
```bash
python src/evaluation.py
```


## Experiments and Results
- **Random Forest**: 100% accuracy on small test set (requires larger validation).
- **Custom CNN**: Overfitting observed during training.
- **MobileNetV2**: 95% validation accuracy with balanced precision/recall.



## Report
A full report with methodology, results (training curves, confusion matrices), and deployment guidelines is available in [`Documentation.pdf`](Documentation.pdf).

