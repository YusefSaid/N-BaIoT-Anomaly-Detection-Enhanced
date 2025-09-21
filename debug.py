import pandas as pd
import numpy as np
from custom_data_loader import NBaloTDataLoader

def debug_data():
    """Debug the dataset to find potential issues"""
    loader = NBaloTDataLoader()
    
    # Load data for debugging
    device = 'danmini_doorbell'
    train_df, test_df = loader.load_device_data(device)
    
    print("Data Debugging Report")
    print("=" * 50)
    
    # Get feature columns (exclude label columns)
    feature_cols = [col for col in train_df.columns if col not in ['label', 'attack_type']]
    X = train_df[feature_cols]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Check for problematic values
    print("\nData Quality Checks:")
    print("-" * 30)
    
    # Check for NaN values
    nan_count = X.isnull().sum().sum()
    print(f"NaN values: {nan_count}")
    
    # Check for infinite values
    inf_count = np.isinf(X.values).sum()
    print(f"Infinite values: {inf_count}")
    
    # Check data ranges
    print(f"\nData Range Analysis:")
    print(f"Minimum value: {X.values.min():.6f}")
    print(f"Maximum value: {X.values.max():.6f}")
    print(f"Mean: {X.values.mean():.6f}")
    print(f"Std: {X.values.std():.6f}")
    
    # Check for extremely large values
    large_values = np.abs(X.values) > 1e6
    print(f"Values > 1M: {large_values.sum()}")
    
    # Check data types
    print(f"\nData Types:")
    print(X.dtypes.value_counts())
    
    # Show some statistics per column
    print(f"\nPer-column statistics (first 10 features):")
    stats = X.iloc[:, :10].describe()
    print(stats)
    
    # Check for columns with zero variance
    zero_var_cols = X.columns[X.var() == 0].tolist()
    if zero_var_cols:
        print(f"\nColumns with zero variance: {len(zero_var_cols)}")
        print(f"Examples: {zero_var_cols[:5]}")
    
    # Check for highly correlated features
    corr_matrix = X.corr().abs()
    high_corr = (corr_matrix > 0.95) & (corr_matrix < 1.0)
    high_corr_pairs = high_corr.sum().sum() // 2
    print(f"\nHighly correlated feature pairs (>0.95): {high_corr_pairs}")
    
    return X, feature_cols

if __name__ == "__main__":
    X, feature_cols = debug_data()