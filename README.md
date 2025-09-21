# N-BaIoT-Anomaly-Detection-Enhanced

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

Right now, we are investigating feature selection, alternative ML models, and cross-device generalization for IoT botnet detection.