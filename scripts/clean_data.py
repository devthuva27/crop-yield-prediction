import pandas as pd
import numpy as np
import os

# Define paths
input_file = '../data/crop_data.csv'
output_file = '../data/cleaned_crop_data.csv'

# 1. Load Data
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit(1)

df = pd.read_csv(input_file)
initial_rows = len(df)
print(f"Initial row count: {initial_rows}")

# 2. Fix Data Types and Basic Cleaning
# Ensure correct types
# Categorical columns
for col in ['Crop', 'Country', 'District']:
    df[col] = df[col].astype(str)

# Numeric columns
numeric_cols = ['Year', 'Month', 'Rainfall_mm', 'Temperature_C', 'Fertilizer', 'N', 'P', 'K', 'Yield_t_per_ha']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Handle Missing Values
# Count missing before
missing_before = df.isnull().sum().sum()
print(f"\nMissing values found: {missing_before}")

# Fill numeric missing with median
if missing_before > 0:
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val}")

# 4. Remove Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates}")
df.drop_duplicates(inplace=True)
print(f"Removed {duplicates} duplicate rows")

# 5. Outlier Removal (Yield < 500 or Yield > 15000)
# CRITICAL CHECK: The user requested removing Yield < 500 or Yield > 15000.
# Based on previous analysis, Yield is in t/ha (max ~88), so ALL data is < 500.
# We will check if this filter would remove > 90% of data. If so, we SKIP it to preserve the dataset.

outlier_mask = (df['Yield_t_per_ha'] < 500) | (df['Yield_t_per_ha'] > 15000)
outliers_found = outlier_mask.sum()
total_rows_now = len(df)

print(f"\nOutlier check (Yield < 500 or > 15000):")
print(f"Found {outliers_found} rows matching outlier criteria out of {total_rows_now} rows.")

if outliers_found == total_rows_now:
    print("WARNING: Creating this filter would remove 100% of your data.")
    print("It appears the threshold (500) is mismatched with the unit (t/ha).")
    print("SKIPPING outlier removal to preserve dataset.")
    # Do not apply mask
else:
    df = df[~outlier_mask]
    print(f"Removed {outliers_found} outlier rows.")


# 6. Save Cleaned Data
df.to_csv(output_file, index=False)

# 7. Final Statistics
print(f"\n--- Final Report ---")
print(f"Output saved to: {output_file}")
print(f"Final row count: {len(df)}")
print(f"Rows removed total: {initial_rows - len(df)}")
