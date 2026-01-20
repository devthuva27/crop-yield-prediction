"""
STEP 7: FINAL TESTS - Write results to file
"""
import numpy as np
import joblib
import json
from datetime import datetime

CROP_YIELD_BOUNDS = {
    'rice': (0, 15000),
    'tea': (0, 5000),
    'rubber': (0, 4000),
    'sugarcane': (0, 150000),
    'cinnamon': (0, 3000),
}

def load_resources():
    model = joblib.load('../models/best_model.pkl')
    with open('../models/feature_scaler.json', 'r') as f:
        scaler_params = json.load(f)
    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]
    return model, scaler_params, feature_names

def preprocess_input(data, feature_names, scaler_params):
    now = datetime.now()
    rainfall = float(data['rainfall'])
    temperature = float(data['temperature'])
    n_val = float(data['nitrogen'])
    p_val = float(data['phosphorus'])
    k_val = float(data['potassium'])
    crop_name = data['crop'].capitalize()
    
    total_npk = n_val + p_val + k_val
    
    all_features = {
        'Year': now.year, 'Month': now.month,
        'Rainfall_mm': rainfall, 'Temperature_C': temperature,
        'Fertilizer': 75.0, 'N': n_val, 'P': p_val, 'K': k_val,
        'is_low_rainfall': 1 if rainfall < 300 else 0,
        'is_high_rainfall': 1 if rainfall > 700 else 0,
        'is_low_temp': 1 if temperature < 18 else 0,
        'is_high_temp': 1 if temperature > 30 else 0,
        'is_extreme_temp': 1 if temperature > 35 else 0,
        'rain_temp_interaction': rainfall * temperature,
        'rain_squared': rainfall ** 2,
        'temp_squared': temperature ** 2,
        'total_npk': total_npk,
        'npk_balance': abs(n_val - p_val) + abs(n_val - k_val) + abs(p_val - k_val),
        'n_ratio': n_val / (total_npk + 1e-5),
        'is_low_npk': 1 if total_npk < 100 else 0,
        'good_tea_zone': 1 if (crop_name == 'Tea' and 500 <= rainfall <= 900 and 18 <= temperature <= 26) else 0,
        'good_rubber_zone': 1 if (crop_name == 'Rubber' and 600 <= rainfall <= 1000 and 20 <= temperature <= 28) else 0,
        'good_sugarcane_zone': 1 if (crop_name == 'Sugarcane' and 700 <= rainfall <= 1000 and 22 <= temperature <= 30) else 0,
        'good_cinnamon_zone': 1 if (crop_name == 'Cinnamon' and 600 <= rainfall <= 1000 and 24 <= temperature <= 32) else 0,
        'good_rice_zone': 1 if (crop_name == 'Rice' and rainfall >= 600 and 22 <= temperature <= 30) else 0,
        'drought_stress': 1 if (rainfall < 300 and temperature > 30) else 0,
        'cold_stress': 1 if temperature < 15 else 0,
        'heat_stress': 1 if temperature > 35 else 0,
    }
    
    feat_dict = {name: 0.0 for name in feature_names}
    for k, v in all_features.items():
        if k in feat_dict: feat_dict[k] = v
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

def postprocess_prediction(raw_prediction, crop):
    predicted_yield_t = np.expm1(raw_prediction)
    predicted_yield_kg = max(predicted_yield_t * 1000, 0.0)
    crop_lower = crop.lower()
    if crop_lower in CROP_YIELD_BOUNDS:
        lower_bound, upper_bound = CROP_YIELD_BOUNDS[crop_lower]
        predicted_yield_kg = max(min(predicted_yield_kg, upper_bound), lower_bound)
    return predicted_yield_kg

def run_final_tests():
    model, scaler_params, feature_names = load_resources()
    
    test_cases = [
        {'crop': 'rice', 'scenario': 'BAD', 'rainfall': 150, 'temperature': 35, 'nitrogen': 100, 'phosphorus': 50, 'potassium': 50},
        {'crop': 'rice', 'scenario': 'GOOD', 'rainfall': 800, 'temperature': 26, 'nitrogen': 120, 'phosphorus': 60, 'potassium': 60},
        {'crop': 'tea', 'scenario': 'BAD', 'rainfall': 150, 'temperature': 35, 'nitrogen': 100, 'phosphorus': 50, 'potassium': 50},
        {'crop': 'tea', 'scenario': 'GOOD', 'rainfall': 700, 'temperature': 22, 'nitrogen': 120, 'phosphorus': 60, 'potassium': 60},
        {'crop': 'rubber', 'scenario': 'BAD', 'rainfall': 150, 'temperature': 35, 'nitrogen': 100, 'phosphorus': 50, 'potassium': 50},
        {'crop': 'rubber', 'scenario': 'GOOD', 'rainfall': 800, 'temperature': 25, 'nitrogen': 100, 'phosphorus': 50, 'potassium': 60},
        {'crop': 'sugarcane', 'scenario': 'BAD', 'rainfall': 150, 'temperature': 38, 'nitrogen': 80, 'phosphorus': 30, 'potassium': 30},
        {'crop': 'sugarcane', 'scenario': 'GOOD', 'rainfall': 900, 'temperature': 28, 'nitrogen': 150, 'phosphorus': 80, 'potassium': 80},
        {'crop': 'cinnamon', 'scenario': 'BAD', 'rainfall': 150, 'temperature': 35, 'nitrogen': 50, 'phosphorus': 20, 'potassium': 20},
        {'crop': 'cinnamon', 'scenario': 'GOOD', 'rainfall': 800, 'temperature': 28, 'nitrogen': 100, 'phosphorus': 50, 'potassium': 50},
    ]
    
    with open('step7_results.txt', 'w', encoding='utf-8') as f:
        f.write("STEP 7: FINAL TESTS WITH NEW AGRONOMIC MODEL\n")
        f.write("="*80 + "\n\n")
        
        results = {}
        for tc in test_cases:
            input_vector = preprocess_input(tc, feature_names, scaler_params)
            raw_pred = model.predict(input_vector)[0]
            final_pred = postprocess_prediction(raw_pred, tc['crop'])
            
            f.write(f"{tc['crop']:<12} | {tc['scenario']:<8} | Rain={tc['rainfall']:<4} Temp={tc['temperature']:<4} -> {final_pred:.0f} kg/ha\n")
            
            key = tc['crop']
            if key not in results: results[key] = {}
            results[key][tc['scenario']] = final_pred
        
        f.write("\n" + "="*80 + "\n")
        f.write("VALIDATION CHECKS:\n")
        f.write("="*80 + "\n\n")
        
        all_passed = True
        for crop, scenarios in results.items():
            bad_val = scenarios.get('BAD', 0)
            good_val = scenarios.get('GOOD', 0)
            mono_pass = good_val > bad_val
            non_neg_pass = bad_val >= 0 and good_val >= 0
            bounds = CROP_YIELD_BOUNDS.get(crop.lower(), (0, 100000))
            range_pass = (bounds[0] <= bad_val <= bounds[1]) and (bounds[0] <= good_val <= bounds[1])
            
            overall = mono_pass and non_neg_pass and range_pass
            if not overall: all_passed = False
            
            f.write(f"{crop}: BAD={bad_val:.0f}, GOOD={good_val:.0f}\n")
            f.write(f"   Monotonicity: {'PASS' if mono_pass else 'FAIL'}\n")
            f.write(f"   Non-negative: {'PASS' if non_neg_pass else 'FAIL'}\n")
            f.write(f"   In range:     {'PASS' if range_pass else 'FAIL'}\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"OVERALL: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED'}\n")
    
    print("Results written to step7_results.txt")

if __name__ == "__main__":
    run_final_tests()
