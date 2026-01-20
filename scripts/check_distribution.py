
import pandas as pd
import numpy as np

def check_data():
    df = pd.read_csv('../data/crop_data.csv')
    
    print("--- Basic Stats of Target (Yield_t_per_ha) ---")
    print(df['Yield_t_per_ha'].describe())
    
    print("\n--- Stats by Crop ---")
    crop_stats = df.groupby('Crop')['Yield_t_per_ha'].agg(['count', 'min', 'max', 'mean', 'median', 'std'])
    print(crop_stats)
    
    negative_yields = df[df['Yield_t_per_ha'] < 0]
    if not negative_yields.empty:
        print(f"\nFound {len(negative_yields)} rows with negative yields!")
        print(negative_yields)
    else:
        print("\nNo negative yields found in original dataset.")

if __name__ == "__main__":
    check_data()
