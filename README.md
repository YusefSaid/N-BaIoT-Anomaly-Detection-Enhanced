# N-BaIoT Enhanced Research

Research project extending IoT botnet detection using the N-BaIoT dataset with improved methodology and additional ML approaches.

## Based on

- **Original Paper**: Meidan et al. (2018) "N-BaIoT—Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders"
- **Code Base**: [husseinalygit/N-BaIoT-reloaded](https://github.com/husseinalygit/N-BaIoT-reloaded)

## Current Status

- Fixed autoencoder methodology (training on benign traffic only)
- Achieved 80.49% F1 score vs 17.69% with original approach
- Implemented proper data validation and preprocessing
- Working on additional ML models and cross-device evaluation

## Dataset

**Important**: This project uses a different dataset than the Hussein repository.

- **Hussein's repo**: Uses the original 2018 N-BaIoT dataset (1.7GB, raw network captures)
- **This project**: Uses the 2021 preprocessed dataset from [Mendeley Data](https://data.mendeley.com/datasets/hpbszmrns7/1)

The preprocessed dataset (455MB) was created by Gutowska (2021) for anomaly detection benchmarking, featuring:
- Balanced benign/attack ratios (unlike the original)
- Pre-extracted statistical features 
- Cleaned and resampled data from 5 IoT devices
- Ready for machine learning without additional preprocessing

Download and place in `data/` directory.

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scikit-learn tensorflow matplotlib

# Validate data
python custom_data_loader.py

# Train autoencoder
python custom_model.py
```

## Files

- `custom_data_loader.py` - Enhanced data loading with proper methodology
- `custom_model.py` - Autoencoder with threshold optimization  
- `original_code/` - Original implementation files
- `papers/` - Research papers and references

## Research Notes

<<<<<<< HEAD
Currently investigating feature selection, alternative ML models, and cross-device generalization for IoT botnet detection.
=======
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
