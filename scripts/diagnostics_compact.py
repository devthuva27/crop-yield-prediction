
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

def run_diagnostics():
    model = joblib.load('../models/xgboost_model.pkl')
    with open('../models/feature_scaler.json', 'r') as f:
        scaler_params = json.load(f)
    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]

    df = pd.read_csv('../data/crop_data.csv')
    
    crops = ['Rice', 'Sugarcane', 'Tea', 'Rubber', 'Cinnamon']
    
    print(f"{'Crop':<12} | {'Actual':<10} | {'Pred (t/ha)':<12} | {'Pred (kg/ha)':<12}")
    print("-" * 55)
    
    for crop in crops:
        row = df[df['Crop'] == crop].iloc[0].to_dict()
        
        # Preprocess
        raw_features = {
            'Year': 2024, 'Month': 5,
            'Rainfall_mm': float(row['Rainfall_mm']),
            'Temperature_C': float(row['Temperature_C']),
            'Fertilizer': 75.0,
            'N': float(row['N']), 'P': float(row['P']), 'K': float(row['K'])
        }
        raw_features['rainfall_to_temperature_ratio'] = raw_features['Rainfall_mm'] / (raw_features['Temperature_C'] + 1e-5)
        raw_features['total_nutrients'] = raw_features['N'] + raw_features['P'] + raw_features['K']
        raw_features['avg_nutrient'] = raw_features['total_nutrients'] / 3.0
        raw_features['nutrient_balance'] = abs(raw_features['N'] - raw_features['P']) + abs(raw_features['P'] - raw_features['K'])
        
        feat_dict = {name: 0.0 for name in feature_names}
        for k, v in raw_features.items():
            if k in feat_dict: feat_dict[k] = v
        crop_col = f"Crop_{crop}"
        if crop_col in feat_dict: feat_dict[crop_col] = 1.0

        scaled_vector = []
        for name in feature_names:
            val = feat_dict[name]
            if name in scaler_params:
                s_min, s_max = scaler_params[name]['min'], scaler_params[name]['max']
                scaled_val = (val - s_min) / (s_max - s_min) if s_max - s_min != 0 else 0.0
                scaled_vector.append(scaled_val)
            else:
                scaled_vector.append(val)
        
        prediction = model.predict(np.array([scaled_vector]))[0]
        actual = row['Yield_t_per_ha']
        
        print(f"{crop:<12} | {actual:<10.2f} | {prediction:<12.4f} | {prediction*1000:<12.2f}")

if __name__ == "__main__":
    run_diagnostics()
