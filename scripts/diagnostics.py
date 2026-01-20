
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Mock Config
class Config:
    MODEL_PATH = '../models/xgboost_model.pkl'
    SCALER_PATH = '../models/feature_scaler.json'
    FEATURE_NAMES_PATH = 'feature_names.txt'

def preprocess_and_enc(data, feature_names, scaler_params):
    now = datetime.now()
    raw_features = {
        'Year': now.year,
        'Month': now.month,
        'Rainfall_mm': float(data['Rainfall_mm']),
        'Temperature_C': float(data['Temperature_C']),
        'Fertilizer': 75.0,
        'N': float(data['N']),
        'P': float(data['P']),
        'K': float(data['K'])
    }
    
    raw_features['rainfall_to_temperature_ratio'] = raw_features['Rainfall_mm'] / (raw_features['Temperature_C'] + 1e-5)
    raw_features['total_nutrients'] = raw_features['N'] + raw_features['P'] + raw_features['K']
    raw_features['avg_nutrient'] = raw_features['total_nutrients'] / 3.0
    raw_features['nutrient_balance'] = abs(raw_features['N'] - raw_features['P']) + abs(raw_features['P'] - raw_features['K'])
    
    feat_dict = {name: 0.0 for name in feature_names}
    for k, v in raw_features.items():
        if k in feat_dict:
            feat_dict[k] = v
            
    crop_col = f"Crop_{data['Crop']}"
    if crop_col in feat_dict:
        feat_dict[crop_col] = 1.0

    scaled_vector = []
    for name in feature_names:
        val = feat_dict[name]
        if name in scaler_params:
            s_min = scaler_params[name]['min']
            s_max = scaler_params[name]['max']
            if s_max - s_min == 0:
                scaled_val = 0.0
            else:
                scaled_val = (val - s_min) / (s_max - s_min)
            scaled_vector.append(scaled_val)
        else:
            scaled_vector.append(val)
            
    return np.array([scaled_vector])

def run_diagnostics():
    # Load resources
    model = joblib.load(Config.MODEL_PATH)
    with open(Config.SCALER_PATH, 'r') as f:
        scaler_params = json.load(f)
    with open(Config.FEATURE_NAMES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]

    df = pd.read_csv('../data/crop_data.csv')
    
    crops = ['Rice', 'Sugarcane', 'Tea', 'Rubber', 'Cinnamon']
    results = []
    
    for crop in crops:
        # Take first row of this crop
        row = df[df['Crop'] == crop].iloc[0].to_dict()
        input_vector = preprocess_and_enc(row, feature_names, scaler_params)
        prediction = model.predict(input_vector)[0]
        actual = row['Yield_t_per_ha']
        
        results.append({
            'Crop': crop,
            'Actual (t/ha)': actual,
            'Predicted (t/ha)': prediction,
            'Predicted (kg/ha)': prediction * 1000
        })
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string())

if __name__ == "__main__":
    run_diagnostics()
