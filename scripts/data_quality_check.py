
import pandas as pd
import numpy as np
import os

# Paths
train_path = '../data/processed_train.csv'
test_path = '../data/processed_test.csv'
report_path = 'data_quality_report.txt'

def check_data_quality():
    report_lines = []
    score = 100
    
    # 1. Load Data
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Files not found.")
        return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    report_lines.append("=== Comprehensive Data Quality Report ===\n")
    
    # 2. Statistics
    report_lines.append("1. Dataset Statistics")
    report_lines.append(f"   Training Set Shape: {df_train.shape[0]} rows x {df_train.shape[1]} columns")
    report_lines.append(f"   Test Set Shape:     {df_test.shape[0]} rows x {df_test.shape[1]} columns")
    
    mem_train = df_train.memory_usage(deep=True).sum() / 1024
    mem_test = df_test.memory_usage(deep=True).sum() / 1024
    report_lines.append(f"   Memory Usage:       {mem_train:.2f} KB (Train) + {mem_test:.2f} KB (Test)")
    report_lines.append("")

    # 3. Checks
    report_lines.append("2. Quality Checks")
    
    # Check 1: Column Consistency
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    if train_cols == test_cols:
        report_lines.append("   ✓ Column Consistency Check: PASSED")
    else:
        report_lines.append("   X Column Consistency Check: FAILED")
        report_lines.append(f"     Mismatch: {train_cols ^ test_cols}")
        score -= 50

    # Check 2: Missing Values
    missing_train = df_train.isnull().sum().sum()
    missing_test = df_test.isnull().sum().sum()
    if missing_train == 0 and missing_test == 0:
        report_lines.append("   ✓ Missing Values Check:     PASSED")
    else:
        report_lines.append(f"   X Missing Values Check:     FAILED (Train: {missing_train}, Test: {missing_test})")
        score -= 50

    # Check 3: Duplicates
    dup_train = df_train.duplicated().sum()
    dup_test = df_test.duplicated().sum()
    if dup_train == 0 and dup_test == 0:
        report_lines.append("   ✓ Duplicate Rows Check:     PASSED")
    else:
        report_lines.append(f"   X Duplicate Rows Check:     FAILED (Train: {dup_train}, Test: {dup_test})")
        score -= 20

    # Check 4: Data Types
    # Expecting mostly float/int for processed data
    non_numeric_train = df_train.select_dtypes(exclude=[np.number]).shape[1]
    if non_numeric_train == 0:
        report_lines.append("   ✓ Data Type Validity Check: PASSED")
    else:
        report_lines.append(f"   X Data Type Validity Check: FAILED ({non_numeric_train} non-numeric cols)")
        score -= 20

    # Check 5: Range Check
    # Features should be 0-1 (except target). 
    # Target 'Yield_t_per_ha' was kept as is.
    target_col = 'Yield_t_per_ha'
    feature_cols = [c for c in df_train.columns if c != target_col]
    
    range_issues = 0
    
    # Check features are roughly 0-1
    # Allow tiny epsilon for float precision
    min_feats = df_train[feature_cols].min().min()
    max_feats = df_train[feature_cols].max().max()
    
    if min_feats < -0.01 or max_feats > 1.01:
        range_issues += 1
        msg = f"Features outside 0-1 range (Min: {min_feats}, Max: {max_feats})"
    else:
        msg = "Features within 0-1 range"

    # Check target non-negative
    if df_train[target_col].min() < 0:
        range_issues += 1
        msg += "; Target has negative values"
    
    if range_issues == 0:
        report_lines.append("   ✓ Value Range Check:        PASSED")
    else:
        report_lines.append(f"   X Value Range Check:        FAILED ({msg})")
        score -= 20

    # 4. Final Score
    if score < 0: score = 0
    report_lines.append("\n3. Final Assessment")
    report_lines.append(f"   Overall Quality Score: {score}/100")
    
    if score == 100:
        recommendation = "Data is ready for modeling."
    elif score >= 80:
        recommendation = "Data is acceptable but has minor issues (duplicates/ranges)."
    else:
        recommendation = "Data requires fixing before modeling."
        
    report_lines.append(f"   Recommendation: {recommendation}")

    # Write to file with explicit utf-8 encoding to avoid win issues, 
    # but safest is just remove fancy chars or force encoding.
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
    except Exception:
         # Fallback for simple write if encoding fails
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines).replace('✓', '[OK]').replace('X', '[ERR]'))
    
    print(f"Report generated: {report_path}")
    # Print strictly ascii to console to avoid win console crash
    safe_output = "\n".join(report_lines).replace('✓', '[PASS]').replace('X', '[FAIL]')
    print(safe_output)

if __name__ == "__main__":
    check_data_quality()
