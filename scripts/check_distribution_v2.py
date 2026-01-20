
import pandas as pd
import numpy as np
import sys

# Set encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

def check_data():
    df = pd.read_csv('../data/crop_data.csv')
    
    print("--- Stats by Crop ---")
    crop_stats = df.groupby('Crop')['Yield_t_per_ha'].agg(['count', 'min', 'max', 'mean'])
    print(crop_stats.to_string())
    
    print("\n--- Any Negative? ---")
    print(f"Total negatives: {(df['Yield_t_per_ha'] < 0).sum()}")

if __name__ == "__main__":
    check_data()
