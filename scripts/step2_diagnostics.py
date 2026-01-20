"""
STEP 2: Diagnostic function - writes to file for full output
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

def load_model_resources():
    model = joblib.load('../models/best_model.pkl')
    with open('../models/feature_scaler.json', 'r') as f:
        scaler_params = json.load(f)
    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]
    return model, scaler_params, feature_names

def preprocess_input(data, feature_names, scaler_params):
    now = datetime.now()
    raw_features = {
        'Year': now.year, 'Month': now.month,
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
    return np.array([scaled_vector])

def predict_yield(model, input_vector):
    pred_log = model.predict(input_vector)[0]
    pred_t = np.expm1(pred_log)
    pred_kg = max(pred_t * 1000, 0.0)
    return pred_log, pred_t, pred_kg

def run_scenario_diagnostics():
    model, scaler_params, feature_names = load_model_resources()
    crops = ['rice', 'tea', 'rubber', 'sugarcane', 'cinnamon']
    scenarios = {
        'A_LowRain_HighHeat': {'rainfall': 150, 'temperature': 35},
        'B_HighRain_ModHeat': {'rainfall': 800, 'temperature': 24},
        'C_Medium': {'rainfall': 500, 'temperature': 26},
        'D_ExtremeHeat': {'rainfall': 500, 'temperature': 38}
    }
    medium_npk = {'nitrogen': 100, 'phosphorus': 50, 'potassium': 50}
    
    with open('step2_results.txt', 'w', encoding='utf-8') as f:
        f.write("STEP 2: PER-CROP SCENARIO DIAGNOSTICS\n")
        f.write("="*80 + "\n\n")
        
        results = []
        for crop in crops:
            f.write(f"\nCROP: {crop.upper()}\n")
            f.write("-"*40 + "\n")
            for scenario_name, scenario_vals in scenarios.items():
                data = {'crop': crop, **scenario_vals, **medium_npk}
                input_vector = preprocess_input(data, feature_names, scaler_params)
                raw_log, pred_t, pred_kg = predict_yield(model, input_vector)
                f.write(f"  {scenario_name}: Rain={scenario_vals['rainfall']}, Temp={scenario_vals['temperature']} -> {pred_kg:.0f} kg/ha\n")
                results.append({'crop': crop, 'scenario': scenario_name, 'pred_kg': pred_kg})
        
        f.write("\n\nMONOTONICITY CHECK (B should > A):\n")
        f.write("-"*40 + "\n")
        for crop in crops:
            a_pred = [r for r in results if r['crop'] == crop and 'A_' in r['scenario']][0]['pred_kg']
            b_pred = [r for r in results if r['crop'] == crop and 'B_' in r['scenario']][0]['pred_kg']
            status = "PASS" if b_pred > a_pred else "FAIL"
            f.write(f"  {crop}: A={a_pred:.0f}, B={b_pred:.0f} -> {status}\n")
    
    print("Results written to step2_results.txt")

if __name__ == "__main__":
    run_scenario_diagnostics()
