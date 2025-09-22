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

## Technologies

**Core Stack:**
- Python 3.7+, TensorFlow 2.x, scikit-learn, NumPy, Pandas

**ML Techniques:**
- Deep autoencoders for unsupervised anomaly detection
- RobustScaler preprocessing, threshold optimization
- Comprehensive evaluation metrics (F1, precision, recall)

**Hardware:** 8GB+ RAM recommended, GPU optional for faster training

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
- `debug_data.py` - Data quality analysis and issue detection
- `validate_data.py` - Comprehensive dataset validation and consistency checks
- `original_code/` - Original implementation files
- `papers/` - Research papers and references

### Utility Scripts

**When to use debug_data.py:**
- Before training to identify data quality issues
- When working with new datasets
- To understand feature distributions and correlations

**When to use validate_data.py:**
- To verify dataset integrity after download
- To check train/test file consistency
- For comprehensive data quality reporting

## Research Notes

Right now, we are investigating feature selection, alternative ML models, and cross-device generalization for IoT botnet detection.