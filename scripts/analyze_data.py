import pandas as pd
import os

# Define file paths
data_path = '../data/crop_data.csv'
output_file = 'data_analysis.txt'

# Check if data file exists
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

# Read the data
try:
    df = pd.read_csv(data_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Prepare analysis output
output_lines = []

output_lines.append("--- Crop Data Analysis ---")
output_lines.append("\n1. Basic Information")
output_lines.append(f"Rows: {df.shape[0]}")
output_lines.append(f"Columns: {df.shape[1]}")

output_lines.append("\nColumn Names:")
output_lines.append(str(list(df.columns)))

output_lines.append("\n2. First 5 Rows")
output_lines.append(df.head().to_string())

output_lines.append("\n3. Data Types")
output_lines.append(df.dtypes.to_string())

output_lines.append("\n4. Missing Values")
output_lines.append(df.isnull().sum().to_string())

output_lines.append("\n5. Summary Statistics")
output_lines.append(df.describe().to_string())

# Join all lines
full_output = "\n".join(output_lines)

# Print to console
print(full_output)

# Save to file
try:
    with open(output_file, 'w') as f:
        f.write(full_output)
    print(f"\nAnalysis saved to {output_file}")
except Exception as e:
    print(f"Error writing output file: {e}")
