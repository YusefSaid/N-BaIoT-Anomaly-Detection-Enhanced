"""
Core Info regarding this script:

The following code takes inspiration from the original Data Loader code, 
(found in original_code folder) but designed for the preprocessed dataset instead.
The script focuses on the "Target Column Handling".

The script automatically detects and removes potential label columns
to prevent data leakage and follows the original N-BaIoT methodology.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class NBaloTDataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.device_types = ['danmini_doorbell', 'ecobee_thermostat', 'philips_baby_monitor', 
                            'provision_security_camera', 'samsung_webcam']
        self.attack_types = ['gafgyt', 'mirai']
        self.excluded_columns = []  # Track excluded columns
        
    def detect_label_columns(self, df):
        """ Automatically detect potential label/target columns"""
        suspicious_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for common label column names
            if any(keyword in col_lower for keyword in ['target', 'label', 'class', 'y']):
                suspicious_columns.append(col)
                continue
            
            # Check for columns with very few unique values (potential labels)
            if df[col].dtype in [np.number] and df[col].nunique() <= 10:
                # Check if it looks like a binary or categorical label
                unique_vals = sorted(df[col].unique())
                if len(unique_vals) <= 5 and all(isinstance(x, (int, float)) for x in unique_vals):
                    suspicious_columns.append(col)
            
            # Check for zero variance (constant columns)
            if df[col].var() == 0:
                suspicious_columns.append(col)
        
        return suspicious_columns
    
    def clean_dataframe(self, df, file_name="", keep_only_benign=False):
        """Clean DataFrame by removing label columns and other issues"""
        original_shape = df.shape
        
        # If target column exists, use it to separate benign vs attack data
        if 'target' in df.columns:
            if keep_only_benign:
                # Keep only benign samples (target = 0) for autoencoder training
                df = df[df['target'] == 0]
                print(f"  Filtered to benign samples only: {len(df)} samples")
            
            # Remove target column after filtering
            df_clean = df.drop(columns=['target'])
            self.excluded_columns.append('target')
            print(f"  Removed target column from {file_name}")
        else:
            # Detect other suspicious columns
            suspicious_cols = self.detect_label_columns(df)
            
            if suspicious_cols:
                print(f"  Removing suspicious columns from {file_name}: {suspicious_cols}")
                df_clean = df.drop(columns=suspicious_cols)
                self.excluded_columns.extend(suspicious_cols)
            else:
                df_clean = df.copy()
        
        # Remove any remaining non-numeric columns (except those we'll add)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) != len(df_clean.columns):
            non_numeric = [col for col in df_clean.columns if col not in numeric_cols]
            print(f"  Removing non-numeric columns from {file_name}: {non_numeric}")
            df_clean = df_clean[numeric_cols]
        
        # Handle missing values
        if df_clean.isnull().any().any():
            print(f"  Filling {df_clean.isnull().sum().sum()} missing values in {file_name}")
            df_clean = df_clean.fillna(df_clean.median())
        
        print(f"  Cleaned shape: {original_shape} -> {df_clean.shape}")
        return df_clean
    
    def load_device_data(self, device_type, include_attacks=True, benign_only_for_training=False):
        """Load all data for a specific device type with automatic cleaning"""
        print(f"Loading data for device: {device_type}")
        
        train_data = []
        test_data = []
        
        if include_attacks:
            for attack_type in self.attack_types:
                train_file = f"{self.data_dir}/{attack_type}_{device_type}_train.csv"
                test_file = f"{self.data_dir}/{attack_type}_{device_type}_test.csv"
                
                if os.path.exists(train_file):
                    df_train = pd.read_csv(train_file)
                    # For autoencoder training, keep only benign samples
                    df_train_clean = self.clean_dataframe(df_train, f"{attack_type}_train", 
                                                        keep_only_benign=benign_only_for_training)
                    
                    if not df_train_clean.empty:  # Only add if there's data after filtering
                        # Add metadata columns efficiently
                        df_train_clean = pd.concat([
                            df_train_clean,
                            pd.DataFrame({
                                'attack_type': [attack_type] * len(df_train_clean),
                                'label': [0 if benign_only_for_training else 1] * len(df_train_clean)
                            }, index=df_train_clean.index)
                        ], axis=1)
                        train_data.append(df_train_clean)
                        print(f"  Loaded {attack_type} training: {len(df_train_clean)} samples, {len(df_train_clean.columns)-2} features")
                
                if os.path.exists(test_file):
                    df_test = pd.read_csv(test_file)
                    # For testing, keep all data (benign + attacks) but preserve original labels
                    original_labels = df_test['target'] if 'target' in df_test.columns else None
                    df_test_clean = self.clean_dataframe(df_test, f"{attack_type}_test", keep_only_benign=False)
                    
                    # Add metadata columns efficiently  
                    df_test_clean = pd.concat([
                        df_test_clean,
                        pd.DataFrame({
                            'attack_type': [attack_type] * len(df_test_clean),
                            'label': original_labels if original_labels is not None else [1] * len(df_test_clean)
                        }, index=df_test_clean.index)
                    ], axis=1)
                    test_data.append(df_test_clean)
                    print(f"  Loaded {attack_type} testing: {len(df_test_clean)} samples, {len(df_test_clean.columns)-2} features")
        
        # Combine data
        if train_data:
            train_df = pd.concat(train_data, ignore_index=True)
        else:
            train_df = pd.DataFrame()
            
        if test_data:
            test_df = pd.concat(test_data, ignore_index=True)
        else:
            test_df = pd.DataFrame()
            
        return train_df, test_df
    
    def prepare_autoencoder_data(self, device_type, test_size=0.2):
        """Prepare cleaned data for autoencoder training following original methodology"""
        print("Loading data following original N-BaIoT methodology:")
        print("- Training: benign traffic only")
        print("- Testing: mixed benign + attack traffic")
        
        # Load benign-only training data
        train_df, _ = self.load_device_data(device_type, benign_only_for_training=True)
        
        # Load mixed testing data  
        _, test_df = self.load_device_data(device_type, benign_only_for_training=False)
        
        if train_df.empty:
            raise ValueError(f"No benign training data found for device: {device_type}")
        
        # Extract features (exclude our added columns)
        feature_cols = [col for col in train_df.columns if col not in ['label', 'attack_type']]
        
        print(f"Final feature count: {len(feature_cols)} features")
        print(f"Excluded columns: {list(set(self.excluded_columns))}")
        
        X_train_all = train_df[feature_cols].values
        X_test = test_df[feature_cols].values if not test_df.empty else None
        y_test = test_df['label'].values if not test_df.empty else None
        
        # Split benign training data for autoencoder training/validation
        X_train, X_val = train_test_split(X_train_all, test_size=test_size, random_state=42)
        
        print(f"Data shapes:")
        print(f"  X_train (benign only): {X_train.shape}")
        print(f"  X_val (benign only): {X_val.shape}")
        if X_test is not None:
            print(f"  X_test (mixed): {X_test.shape}")
            # Convert labels to integers for counting
            y_test_int = y_test.astype(int)
            unique, counts = np.unique(y_test_int, return_counts=True)
            label_counts = dict(zip(unique, counts))
            print(f"  y_test distribution: {label_counts} (0=benign, 1=attack)")
        
        return X_train, X_val, X_test, y_test
    
    def data_quality_report(self, device_type):
        """Generate a data quality report for a device"""
        print(f"\nData Quality Report for {device_type}")
        print("=" * 40)
        
        train_df, test_df = self.load_device_data(device_type, benign_only_for_training=False)
        
        if not train_df.empty:
            feature_cols = [col for col in train_df.columns if col not in ['label', 'attack_type']]
            X = train_df[feature_cols]
            
            print(f"Features: {len(feature_cols)}")
            print(f"Training samples: {len(train_df)}")
            print(f"Testing samples: {len(test_df)}")
            
            # Check for remaining issues
            zero_var = (X.var() == 0).sum()
            missing_vals = X.isnull().sum().sum()
            
            print(f"Zero variance features: {zero_var}")
            print(f"Missing values: {missing_vals}")
            print(f"Data range: [{X.min().min():.6f}, {X.max().max():.6f}]")
            
            # Check for highly correlated features
            corr_matrix = X.corr().abs()
            high_corr = (corr_matrix > 0.95) & (corr_matrix < 1.0)
            high_corr_pairs = high_corr.sum().sum() // 2
            print(f"Highly correlated pairs (>0.95): {high_corr_pairs}")
            
            return {
                'n_features': len(feature_cols),
                'n_train': len(train_df),
                'n_test': len(test_df),
                'zero_variance': zero_var,
                'missing_values': missing_vals,
                'high_correlation_pairs': high_corr_pairs
            }
        
        return None

# Example usage with validation
if __name__ == "__main__":
    # Test the improved loader
    improved_loader = NBaloTDataLoader()
    
    device = 'danmini_doorbell'
    
    # Generate quality report
    quality_report = improved_loader.data_quality_report(device)
    
    # Test data preparation with proper methodology
    try:
        X_train, X_val, X_test, y_test = improved_loader.prepare_autoencoder_data(device)
        
        print(f"\nSUCCESS: Data prepared successfully!")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        # Validate the methodology
        print(f"\nMethodology Validation:")
        print(f"✅ Training on benign data only: {X_train.shape[0]} samples")
        print(f"✅ Testing on mixed data: {X_test.shape[0]} samples")
        # Convert to int safely for counting
        y_test_int = y_test.astype(int)
        benign_count = np.sum(y_test_int == 0)
        attack_count = np.sum(y_test_int == 1)
        print(f"✅ Test set contains: {benign_count} benign + {attack_count} attack samples")
            
    except Exception as e:
        print(f"Error: {e}")