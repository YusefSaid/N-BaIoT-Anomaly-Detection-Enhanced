# N-BaIoT Enhanced: IoT Botnet Attack Detection Research

This repository contains an enhanced implementation and research extension of the N-BaIoT (Network-Based IoT Botnet Attack Detection) project, building upon the original work by Meidan et al. (2018).

## Overview

This project implements and extends IoT botnet detection using deep autoencoders and other machine learning approaches, with proper data preprocessing and methodological improvements for anomaly detection research.

## Key Features

- **Proper Anomaly Detection Methodology**: Trains on benign traffic only, tests on mixed traffic
- **Data Quality Validation**: Automatic detection and removal of label leakage
- **Multiple ML Approaches**: Autoencoders, tree-based methods, and supervised learning
- **Threshold Optimization**: Comprehensive evaluation across different threshold strategies
- **Clean Code Architecture**: Modular design for easy experimentation

## Performance Highlights

Current autoencoder implementation achieves:
- **F1 Score**: 80.49%
- **Precision**: 91.63%
- **Recall**: 71.76%
- **TNR**: 99.65% (excellent at identifying benign traffic)

## Dataset

This project uses the preprocessed N-BaIoT dataset available from Mendeley Data:

**Download Link**: [N-BaIoT Processed Dataset](https://data.mendeley.com/datasets/hpbszmrns7/1)

The dataset contains:
- **Size**: 455 MB (preprocessed and balanced)
- **Devices**: 5 IoT devices (doorbell, thermostat, baby monitor, security camera, webcam)
- **Features**: 115 statistical features extracted from network traffic
- **Attacks**: Mirai and Gafgyt botnet families
- **Format**: Balanced benign/attack ratios for proper evaluation

### Dataset Setup

1. Download the dataset from the link above
2. Extract to a `data/` directory in the project root
3. The structure should be:
```
data/
├── gafgyt_danmini_doorbell_train.csv
├── gafgyt_danmini_doorbell_test.csv
├── mirai_danmini_doorbell_train.csv
├── mirai_danmini_doorbell_test.csv
└── ... (other device files)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/N-BaIoT-reloaded.git
cd N-BaIoT-reloaded

# Install dependencies
pip install numpy pandas scikit-learn tensorflow matplotlib

# Download dataset (see Dataset section above)
```

## Usage

### Quick Start

```bash
# Test data loading and validation
python custom_data_loader.py

# Train autoencoder model
python custom_model.py
```

### Custom Training

```python
from custom_data_loader import NBaloTDataLoader
from custom_model import NBaloTAnomalyDetectorFixed

# Initialize
detector = NBaloTAnomalyDetectorFixed()

# Train on specific device
model, history = detector.train_device_model(
    device_type='danmini_doorbell',
    epochs=50,
    batch_size=128
)
```

## Project Structure

```
├── custom_data_loader.py      # Enhanced data loading with proper methodology
├── custom_model.py            # Autoencoder implementation with threshold optimization
├── debug_data.py             # Data quality validation tools
├── validate_data.py          # Comprehensive dataset validation
├── original_code/            # Original repository files
├── models/                   # Trained model storage (created during training)
├── data/                     # Dataset directory (download separately)
└── README.md                 # This file
```

## Research Improvements

This implementation addresses several methodological issues from the original approach:

### Data Quality
- **Label Leakage Prevention**: Automatic detection and removal of target columns
- **Missing Value Handling**: Robust preprocessing for incomplete data
- **Feature Validation**: Identification of zero-variance and highly correlated features

### Methodology
- **Proper Unsupervised Learning**: Training on benign traffic only
- **Realistic Evaluation**: Testing on mixed benign/attack traffic
- **Threshold Optimization**: Systematic evaluation of detection thresholds
- **Comprehensive Metrics**: Beyond simple accuracy to F1, precision, recall

### Architecture
- **Improved Autoencoder**: Better regularization and stability
- **Robust Scaling**: RobustScaler for outlier handling
- **Gradient Clipping**: Prevention of training instabilities

## Research Extensions

Future work and implemented features:

- [ ] **Tree-based Methods**: Random Forest, XGBoost comparison
- [ ] **Neural Architectures**: CNN, RNN, Transformer implementations  
- [ ] **Explainable AI**: SHAP and LIME interpretation
- [ ] **Cross-device Evaluation**: Generalization across IoT devices
- [ ] **Federated Learning**: Privacy-preserving distributed training
- [ ] **Adversarial Robustness**: Defense against model poisoning

## Results Comparison

| Approach | F1 Score | Precision | Recall | Notes |
|----------|----------|-----------|---------|-------|
| Original Implementation | 17.69% | 100% | 9.70% | Methodological issues |
| **Enhanced Implementation** | **80.49%** | **91.63%** | **71.76%** | **Proper methodology** |

## Citation

If you use this code in your research, please cite both the original work and this implementation:

### Original Paper
```bibtex
@article{meidan2018n,
  title={N-BaIoT—network-based detection of IoT botnet attacks using deep autoencoders},
  author={Meidan, Yair and Bohadana, Michael and Mathov, Yael and Mirsky, Yisroel and Shabtai, Asaf and Breitenbacher, Dominik and Elovici, Yuval},
  journal={IEEE Pervasive Computing},
  volume={17},
  number={3},
  pages={12--22},
  year={2018},
  publisher={IEEE}
}
```

### This Implementation
```bibtex
@misc{nbaiot_enhanced_2024,
  title={N-BaIoT Enhanced: Methodological Improvements for IoT Botnet Detection},
  author={[Your Name]},
  year={2024},
  url={https://github.com/YourUsername/N-BaIoT-reloaded}
}
```

## License

This project builds upon the original N-BaIoT implementation. Please respect the original licensing terms and provide appropriate attribution.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Contact

For questions or collaboration opportunities, please open an issue or contact [yusefsaid@hotmail.com].

---

**Note**: This is a research implementation focused on reproducibility and methodological correctness. Results may vary based on dataset versions and experimental setup.
