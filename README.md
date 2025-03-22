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
- [Deployment](#deployment)
- [Report](#report)
- [License](#license)

## Overview
QR codes are vulnerable to counterfeiting due to easy replication. This project compares original prints with counterfeit versions using:
- **Traditional Approach**: Combines LBP/HOG features with a Random Forest classifier.
- **Deep Learning Approach**: Uses a custom CNN and MobileNetV2 for feature extraction and classification.

*(Insert workflow diagram here if available.)*

## Project Structure
```plaintext
QR_Code_Authentication_Project/
│
├── README.md
├── requirements.txt
├── data/
│   ├── first_print/First Print/     # Original QR codes
│   └── second_print/Second Print/   # Counterfeit QR codes
├── dataset_cnn/                     # Organized CNN dataset
│   ├── train/original/              # Training originals
│   ├── train/counterfeit/           # Training counterfeits
│   ├── validation/original/         # Validation originals
│   └── validation/counterfeit/      # Validation counterfeits
├── src/
│   ├── data_preparation.py          # Dataset organization
│   ├── feature_engineering.py       # LBP/HOG extraction
│   ├── model_rf.py                  # Random Forest pipeline
│   ├── model_cnn.py                 # Custom CNN
│   ├── model_transfer.py            # MobileNetV2 transfer learning
│   ├── evaluation.py                # Metrics and plots
│   └── deploy.py                    # Model deployment
├── notebooks/                       # Experimental notebooks
├── models/                          # Saved models
│   ├── qr_authentication_transfer_model.keras
│   ├── qr_authentication_transfer_model.tflite
│   └── rf_qr_authentication_model.joblib
└── report/                          # Final report
    └── QR_Code_Authentication_Report.pdf
```

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/QR_Code_Authentication_Project.git
   cd QR_Code_Authentication_Project
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
  python src/model_rf.py
  ```
- **Custom CNN**:
  ```bash
  python src/model_cnn.py
  ```
- **MobileNetV2 Transfer Learning**:
  ```bash
  python src/model_transfer.py
  ```

### Evaluation
Generate performance metrics and plots:
```bash
python src/evaluation.py
```

### Deployment
Convert/save models for inference:
```bash
python src/deploy.py
```

## Experiments and Results
- **Random Forest**: 100% accuracy on small test set (requires larger validation).
- **Custom CNN**: Overfitting observed during training.
- **MobileNetV2**: 95% validation accuracy with balanced precision/recall.

*(See `report/QR_Code_Authentication_Report.pdf` for details.)*

## Report
A full report with methodology, results (training curves, confusion matrices), and deployment guidelines is available in [`report/QR_Code_Authentication_Report.pdf`](report/QR_Code_Authentication_Report.pdf).

## License
MIT License. See [LICENSE](LICENSE).
