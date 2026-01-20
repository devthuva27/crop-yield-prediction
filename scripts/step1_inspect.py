"""
STEP 1: Inspect current pipeline and data
"""
import pandas as pd
import numpy as np

def inspect_data():
    print("="*60)
    print("STEP 1: INSPECTING CURRENT PIPELINE")
    print("="*60)
    
    # Load raw data
    df = pd.read_csv('../data/crop_data.csv')
    
    print("\n1. DATASET OVERVIEW:")
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    print("\n2. UNIQUE CROPS:")
    print(f"   {df['Crop'].unique()}")
    
    print("\n3. PER-CROP YIELD STATISTICS (t/ha):")
    stats = df.groupby('Crop')['Yield_t_per_ha'].agg(['count', 'min', 'max', 'mean', 'std'])
    print(stats.to_string())
    
    print("\n4. CHECK FOR NEGATIVE YIELDS:")
    neg_yields = df[df['Yield_t_per_ha'] < 0]
    if len(neg_yields) > 0:
        print(f"   WARNING: Found {len(neg_yields)} rows with negative yields!")
        print(neg_yields.groupby('Crop')['Yield_t_per_ha'].agg(['count', 'min']))
    else:
        print("   ✓ No negative yields in raw data")
    
    print("\n5. FEATURE RANGES:")
    feature_cols = ['Rainfall_mm', 'Temperature_C', 'N', 'P', 'K']
    for col in feature_cols:
        print(f"   {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    print("\n6. YIELD RANGES BY CROP (converted to kg/ha):")
    for crop in df['Crop'].unique():
        crop_data = df[df['Crop'] == crop]['Yield_t_per_ha']
        print(f"   {crop}: {crop_data.min()*1000:.0f} - {crop_data.max()*1000:.0f} kg/ha (mean: {crop_data.mean()*1000:.0f})")
    
    return df

if __name__ == "__main__":
    inspect_data()
