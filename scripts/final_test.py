
import pandas as pd
import numpy as np
import joblib
import json

def test_final_pipeline():
    # Load resources
    model = joblib.load('../models/best_model.pkl')
    with open('../models/feature_scaler.json', 'r') as f:
        scaler_params = json.load(f)
    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]

    crops = ['Rice', 'Sugarcane', 'Tea', 'Rubber', 'Cinnamon']
    # Use realistic median values for inputs
    test_inputs = [
        {'crop': 'Rice', 'rainfall': 450, 'temperature': 25, 'nitrogen': 80, 'phosphorus': 25, 'potassium': 20},
        {'crop': 'Sugarcane', 'rainfall': 600, 'temperature': 28, 'nitrogen': 120, 'phosphorus': 40, 'potassium': 50},
        {'crop': 'Tea', 'rainfall': 800, 'temperature': 20, 'nitrogen': 100, 'phosphorus': 30, 'potassium': 40},
        {'crop': 'Rubber', 'rainfall': 1000, 'temperature': 27, 'nitrogen': 60, 'phosphorus': 20, 'potassium': 30},
        {'crop': 'Cinnamon', 'rainfall': 500, 'temperature': 26, 'nitrogen': 50, 'phosphorus': 15, 'potassium': 10}
    ]
    
    print(f"{'Crop':<12} | {'Params (R/T/N/P/K)':<25} | {'Predicted (kg/ha)':<15}")
    print("-" * 60)
    
    for data in test_inputs:
        # Preprocess mimic app.py
        raw_features = {
            'Year': 2026, 'Month': 6,
            'Rainfall_mm': float(data['rainfall']),
            'Temperature_C': float(data['temperature']),
            'Fertilizer': 75.0,
            'N': float(data['nitrogen']), 'P': float(data['phosphorus']), 'K': float(data['potassium'])
        }
        raw_features['rainfall_to_temperature_ratio'] = raw_features['Rainfall_mm'] / (raw_features['Temperature_C'] + 1e-5)
        raw_features['total_nutrients'] = raw_features['N'] + raw_features['P'] + raw_features['K']
        raw_features['avg_nutrient'] = raw_features['total_nutrients'] / 3.0
        raw_features['nutrient_balance'] = abs(raw_features['N'] - raw_features['P']) + abs(raw_features['P'] - raw_features['K'])
        
        feat_dict = {name: 0.0 for name in feature_names}
        for k, v in raw_features.items():
            if k in feat_dict: feat_dict[k] = v
        
        crop_name = data['crop'].capitalize()
        crop_col = f"Crop_{crop_name}"
        if crop_col in feat_dict:
            feat_dict[crop_col] = 1.0

        scaled_vector = []
        for name in feature_names:
            val = feat_dict[name]
            if name in scaler_params:
                s_min, s_max = scaler_params[name]['min'], scaler_params[name]['max']
                scaled_val = (val - s_min) / (s_max - s_min) if s_max - s_min != 0 else 0.0
                scaled_vector.append(scaled_val)
            else:
                scaled_vector.append(val)
        
        # Prediction
        pred_log = model.predict(np.array([scaled_vector]))[0]
        pred_t = np.expm1(pred_log)
        pred_kg = max(pred_t * 1000, 0.0)
        
        params_str = f"{data['rainfall']}/{data['temperature']}/{data['nitrogen']}/{data['phosphorus']}/{data['potassium']}"
        print(f"{data['crop']:<12} | {params_str:<25} | {pred_kg:<15.2f}")

if __name__ == "__main__":
    test_final_pipeline()
