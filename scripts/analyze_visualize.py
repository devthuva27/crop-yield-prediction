
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set backend to avoid GUI issues
plt.switch_backend('Agg')

# Define paths
data_path = '../data/crop_data.csv'
output_dir = '../output/plots'
analysis_file = 'data_exploration.txt'

# Ensure output directory exists (redundant check but safe)
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

df = pd.read_csv(data_path)

# Prepare text output
lines = []
lines.append("--- Crop Data Exploration & Visualization ---")

# 2. Statistics & Correlations
# Filter numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Correlation with Yield
corr_matrix = numeric_df.corr()
yield_corr = corr_matrix['Yield_t_per_ha'].sort_values(ascending=False)

lines.append("\n1. Correlation with Yield:")
lines.append(yield_corr.to_string())

# Summary Statistics
lines.append("\n2. Summary Statistics:")
lines.append(df.describe().to_string())

# Grouped Statistics by Crop
lines.append("\n3. Grouped Statistics by Crop (Mean Yield):")
crop_stats = df.groupby('Crop')['Yield_t_per_ha'].describe()
lines.append(crop_stats.to_string())

# 3. Data Quality Check
# Missing Values
missing_values = df.isnull().sum()
lines.append("\n4. Missing Values Count:")
lines.append(missing_values.to_string())

# Outliers
# User criteria: yield < 500 or yield > 15000
# Note: Based on previous analysis, yield is in tonnes/ha (max ~88).
# This criteria might flag all data points if interpreted strictly, 
# or might be intended for kg/ha. We proceed as requested.
outliers = df[(df['Yield_t_per_ha'] < 500) | (df['Yield_t_per_ha'] > 15000)]
num_outliers = len(outliers)
lines.append(f"\n5. Outliers Analysis (Criteria: Yield < 500 or Yield > 15000):")
lines.append(f"Number of outliers found: {num_outliers}")
lines.append(f"Percentage of data flagged as outliers: {(num_outliers/len(df))*100:.2f}%")
if num_outliers == len(df):
    lines.append("Note: All data points were flagged. The threshold criteria might be irrelevant for the unit 't/ha'.")

# Save Analysis to File
with open(analysis_file, 'w') as f:
    f.write("\n".join(lines))
print(f"Analysis text saved to {analysis_file}")

# 4. visualizations

# Set style
sns.set_theme(style="whitegrid")

# Histogram: Yield Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Yield_t_per_ha'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Crop Yield (t/ha)')
plt.xlabel('Yield (t/ha)')
plt.ylabel('Frequency')
plt.savefig(f"{output_dir}/yield_histogram.png")
plt.close()

# Scatter: Rainfall vs Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Rainfall_mm', y='Yield_t_per_ha', hue='Crop', alpha=0.6)
plt.title('Rainfall vs Yield')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Yield (t/ha)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/rainfall_vs_yield.png")
plt.close()

# Scatter: Temperature vs Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Temperature_C', y='Yield_t_per_ha', hue='Crop', alpha=0.6)
plt.title('Temperature vs Yield')
plt.xlabel('Temperature (C)')
plt.ylabel('Yield (t/ha)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/temp_vs_yield.png")
plt.close()

# Scatter: Nitrogen vs Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='N', y='Yield_t_per_ha', hue='Crop', alpha=0.6)
plt.title('Nitrogen (N) vs Yield')
plt.xlabel('Nitrogen (N)')
plt.ylabel('Yield (t/ha)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/n_vs_yield.png")
plt.close()

# Heatmap: Correlation
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

print(f"Plots saved to {output_dir}")
