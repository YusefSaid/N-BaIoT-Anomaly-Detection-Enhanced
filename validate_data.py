"""
Data Validation Script for N-BaIoT Dataset

This script validates the dataset structure and identifies potential issues
like target columns, data leakage, and feature quality problems.
"""

import pandas as pd
import numpy as np
import os
import glob

class DataValidator:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
    def validate_all_files(self):
        """Validate all CSV files in the data directory"""
        print("Data Validation Report")
        print("=" * 50)
        
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        issues_found = []
        
        for file_path in csv_files:
            print(f"\nValidating: {os.path.basename(file_path)}")
            print("-" * 30)
            
            try:
                df = pd.read_csv(file_path)
                file_issues = self.validate_single_file(df, file_path)
                if file_issues:
                    issues_found.extend([(os.path.basename(file_path), issue) for issue in file_issues])
            except Exception as e:
                print(f"ERROR reading file: {e}")
                issues_found.append((os.path.basename(file_path), f"Read error: {e}"))
        
        # Summary
        print(f"\n{'='*50}")
        print("VALIDATION SUMMARY")
        print(f"{'='*50}")
        
        if issues_found:
            print("ISSUES FOUND:")
            for filename, issue in issues_found:
                print(f"  {filename}: {issue}")
        else:
            print("No critical issues found!")
            
        return issues_found
    
    def validate_single_file(self, df, file_path):
        """Validate a single DataFrame"""
        issues = []
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for target/label columns
        suspicious_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['target', 'label', 'class', 'y']):
                suspicious_columns.append(col)
        
        if suspicious_columns:
            print(f"⚠️  SUSPICIOUS COLUMNS (possible labels): {suspicious_columns}")
            issues.append(f"Contains potential label columns: {suspicious_columns}")
            
            # Check if these columns have limited unique values
            for col in suspicious_columns:
                unique_vals = df[col].nunique()
                unique_list = df[col].unique()[:10]  # Show first 10 unique values
                print(f"   {col}: {unique_vals} unique values -> {unique_list}")
                
                if unique_vals <= 10:  # Likely a categorical label
                    issues.append(f"Column '{col}' appears to be a categorical label ({unique_vals} unique values)")
        
        # Check for zero variance columns
        zero_var_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].var() == 0:
                zero_var_cols.append(col)
        
        if zero_var_cols:
            print(f"⚠️  ZERO VARIANCE COLUMNS: {zero_var_cols}")
            issues.append(f"Zero variance columns: {zero_var_cols}")
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"⚠️  MISSING VALUES in columns: {missing_cols}")
            for col in missing_cols:
                missing_count = df[col].isnull().sum()
                print(f"   {col}: {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
            issues.append(f"Missing values in: {missing_cols}")
        
        # Check data types
        print(f"Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Check for extremely large values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            max_val = df[numeric_cols].max().max()
            min_val = df[numeric_cols].min().min()
            
            if abs(max_val) > 1e6 or abs(min_val) > 1e6:
                print(f"⚠️  EXTREME VALUES: Range [{min_val:.2e}, {max_val:.2e}]")
                issues.append(f"Extreme values detected: [{min_val:.2e}, {max_val:.2e}]")
        
        # Check for potential index columns
        potential_index_cols = []
        for col in df.columns:
            if df[col].nunique() == len(df):  # Unique values = number of rows
                potential_index_cols.append(col)
        
        if potential_index_cols:
            print(f"💡 POTENTIAL INDEX COLUMNS: {potential_index_cols}")
            # This is informational, not necessarily an issue
        
        return issues
    
    def recommend_preprocessing_steps(self, issues_found):
        """Recommend preprocessing steps based on found issues"""
        print(f"\n{'='*50}")
        print("RECOMMENDED PREPROCESSING STEPS")
        print(f"{'='*50}")
        
        recommendations = []
        
        # Check for label columns
        label_files = [f for f, issue in issues_found if 'label columns' in issue.lower()]
        if label_files:
            recommendations.append("1. REMOVE LABEL COLUMNS: Use .iloc[:, :-1] or explicitly exclude label columns")
            print("1. REMOVE LABEL COLUMNS:")
            print("   - Add .iloc[:, :-1] to your data loading")
            print("   - Or explicitly exclude columns like 'target', 'label', 'class'")
            print("   - This prevents data leakage!")
        
        # Check for zero variance
        zero_var_files = [f for f, issue in issues_found if 'zero variance' in issue.lower()]
        if zero_var_files:
            recommendations.append("2. REMOVE ZERO VARIANCE FEATURES")
            print("2. REMOVE ZERO VARIANCE FEATURES:")
            print("   - These columns provide no information")
            print("   - Remove them: df = df.loc[:, df.var() != 0]")
        
        # Check for missing values
        missing_files = [f for f, issue in issues_found if 'missing values' in issue.lower()]
        if missing_files:
            recommendations.append("3. HANDLE MISSING VALUES")
            print("3. HANDLE MISSING VALUES:")
            print("   - Option 1: Remove rows with missing values")
            print("   - Option 2: Impute missing values (mean, median)")
            print("   - Option 3: Use algorithms that handle missing values")
        
        # Check for extreme values
        extreme_files = [f for f, issue in issues_found if 'extreme values' in issue.lower()]
        if extreme_files:
            recommendations.append("4. HANDLE EXTREME VALUES")
            print("4. HANDLE EXTREME VALUES:")
            print("   - Apply robust scaling (RobustScaler)")
            print("   - Clip extreme values")
            print("   - Check for data entry errors")
        
        return recommendations
    
    def check_data_consistency(self):
        """Check consistency across train/test splits"""
        print(f"\n{'='*50}")
        print("DATA CONSISTENCY CHECK")
        print(f"{'='*50}")
        
        # Get all unique prefixes (device + attack combinations)
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        train_files = [f for f in csv_files if '_train.csv' in f]
        test_files = [f for f in csv_files if '_test.csv' in f]
        
        print(f"Found {len(train_files)} training files and {len(test_files)} test files")
        
        # Check for matching pairs
        train_prefixes = set([os.path.basename(f).replace('_train.csv', '') for f in train_files])
        test_prefixes = set([os.path.basename(f).replace('_test.csv', '') for f in test_files])
        
        missing_test = train_prefixes - test_prefixes
        missing_train = test_prefixes - train_prefixes
        
        if missing_test:
            print(f"⚠️  Training files without test pairs: {missing_test}")
        
        if missing_train:
            print(f"⚠️  Test files without training pairs: {missing_train}")
        
        # Check feature consistency between train/test pairs
        for prefix in train_prefixes & test_prefixes:
            train_file = os.path.join(self.data_dir, f"{prefix}_train.csv")
            test_file = os.path.join(self.data_dir, f"{prefix}_test.csv")
            
            try:
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)
                
                if list(train_df.columns) != list(test_df.columns):
                    print(f"⚠️  Column mismatch in {prefix}:")
                    print(f"     Train: {len(train_df.columns)} cols")
                    print(f"     Test:  {len(test_df.columns)} cols")
                
            except Exception as e:
                print(f"Error checking {prefix}: {e}")

def main():
    """Main validation function"""
    validator = DataValidator()
    
    # Run full validation
    issues = validator.validate_all_files()
    
    # Check consistency
    validator.check_data_consistency()
    
    # Provide recommendations
    validator.recommend_preprocessing_steps(issues)
    
    print(f"\n{'='*50}")
    print("NEXT STEPS:")
    print("1. Update your data loader to exclude label columns")
    print("2. Re-run training with cleaned data")
    print("3. Compare results with previous baseline")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()