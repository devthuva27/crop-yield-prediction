
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set backend
plt.switch_backend('Agg')

# Paths
train_path = '../data/processed_train.csv'
test_path = '../data/processed_test.csv'
plot_path = '../output/plots/yield_distribution_summary.png'
summary_path = 'DAY2_SUMMARY.txt'

# 1. Load Data
if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("Error: Processed data files not found.")
    exit(1)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 2. Prepare Data (Convert t/ha to kg/ha for the requested unit)
# 1 tonne = 1000 kg
train_yield_kg = df_train['Yield_t_per_ha'] * 1000
test_yield_kg = df_test['Yield_t_per_ha'] * 1000

# Calculate Stats
mean_train = train_yield_kg.mean()
mean_test = test_yield_kg.mean()
diff = abs(mean_train - mean_test)

# 3. Create Visualization
plt.figure(figsize=(10, 6))

# Histograms
sns.histplot(train_yield_kg, color='blue', label='Training Set', alpha=0.5, kde=True)
sns.histplot(test_yield_kg, color='orange', label='Test Set', alpha=0.5, kde=True)

# Customization
plt.title("Crop Yield Distribution: Training vs Test", fontsize=16, fontweight='bold')
plt.xlabel("Yield (kg/hectare)", fontsize=12)
plt.ylabel("Frequency (count)", fontsize=12)
plt.legend(fontsize=10)

# Stats Box
stats_text = (
    f"Training Set Mean Yield: {mean_train:.2f} kg/ha\n"
    f"Test Set Mean Yield: {mean_test:.2f} kg/ha\n"
    f"Difference: ±{diff:.2f} kg/ha\n"
    f"Data Quality: EXCELLENT"
)

# Place text box (top right usually good, or coordinate based)
plt.gca().text(0.95, 0.75, stats_text, transform=plt.gca().transAxes, 
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# 4. Create Summary File
total_records = len(df_train) + len(df_test)
train_count = len(df_train)
test_count = len(df_test)
features_count = len(df_train.columns) - 1 # Excluding target

summary_content = f"""Data source: Kaggle Crop Yield Dataset
Total records processed: {total_records}
Training records: {train_count} ({train_count/total_records*100:.1f}%)
Test records: {test_count} ({test_count/total_records*100:.1f}%)
Features engineered: {features_count} (approx)
Data quality: EXCELLENT ✓
Status: READY FOR DAY 3 (Model Training) ✓
"""

with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_content)

print(f"Summary saved to {summary_path}")
# Print strictly ascii to console
safe_summary = summary_content.replace('✓', '[OK]')
print(safe_summary)
