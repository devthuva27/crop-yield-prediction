"""
CROP YIELD PREDICTION API
=========================
Backend Flask API with agronomic domain knowledge.

CHANGES FROM ORIGINAL:
1. Added agronomic feature engineering (good_*_zone, stress indicators, etc.)
2. Added post-processing with non-negative constraint and crop-specific clipping
3. Improved crop name handling
4. Added MIN_YIELD configuration
"""
import os
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": Config.ALLOWED_ORIGINS}})

# ============================================================================
# STEP 6: CROP-SPECIFIC REALISTIC YIELD RANGES (kg/ha)
# These are loose bounds to catch obviously wrong predictions
# ============================================================================
CROP_YIELD_BOUNDS = {
    'rice': (0, 15000),        # 0-15 t/ha
    'tea': (0, 5000),          # 0-5 t/ha
    'rubber': (0, 4000),       # 0-4 t/ha
    'sugarcane': (0, 150000),  # 0-150 t/ha (realistic range for Sri Lanka)
    'cinnamon': (0, 3000),     # 0-3 t/ha
}

# Minimum yield to return (safety floor)
MIN_YIELD_KG = 0.0

# Global variables for model and metadata
model = None
scaler_params = None
feature_names = None
supabase_client: Client = None

def load_resources():
    global model, scaler_params, feature_names, supabase_client
    
    # Load Model
    try:
        if os.path.exists(Config.MODEL_PATH):
            model = joblib.load(Config.MODEL_PATH)
            logger.info(f"Model loaded from {Config.MODEL_PATH}")
        else:
            logger.error(f"Model file not found at {Config.MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

    # Load Scaler Params
    try:
        if os.path.exists(Config.SCALER_PATH):
            with open(Config.SCALER_PATH, 'r') as f:
                scaler_params = json.load(f)
            logger.info(f"Scaler parameters loaded from {Config.SCALER_PATH}")
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")

    # Load Feature Names
    try:
        if os.path.exists(Config.FEATURE_NAMES_PATH):
            with open(Config.FEATURE_NAMES_PATH, 'r') as f:
                feature_names = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Feature names loaded: {len(feature_names)} features")
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")

    # Initialize Supabase
    if Config.SUPABASE_URL and Config.SUPABASE_KEY:
        try:
            supabase_client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.error(f"Error initializing Supabase: {e}")
    else:
        logger.warning("Supabase credentials missing. Database saving will be disabled.")

# Load resources at startup
load_resources()

def validate_inputs(data):
    """Validate prediction inputs against specified ranges."""
    validation_rules = {
        'rainfall': (100, 1000),
        'temperature': (10, 40),
        'nitrogen': (0, 200),
        'phosphorus': (0, 100),
        'potassium': (0, 100)
    }
    
    for field, (min_val, max_val) in validation_rules.items():
        if field not in data:
            return f"Missing required field: {field}"
        
        try:
            val = float(data[field])
            if not (min_val <= val <= max_val):
                return f"Invalid {field} value. Must be between {min_val} and {max_val}."
        except ValueError:
            return f"Invalid type for {field}. Must be a number."
            
    if 'crop' not in data:
        return "Missing required field: crop"
    
    valid_crops = ['rice', 'tea', 'rubber', 'sugarcane', 'cinnamon']
    if data['crop'].lower() not in valid_crops:
        return f"Invalid crop. Must be one of: {valid_crops}"
        
    return None


def preprocess_and_enc(data):
    """
    Transform user input into model feature vector.
    
    IMPORTANT: This function must match the feature engineering done in training!
    See train_with_agronomy.py for the corresponding training code.
    """
    now = datetime.now()
    
    # Extract input values
    rainfall = float(data['rainfall'])
    temperature = float(data['temperature'])
    n_val = float(data['nitrogen'])
    p_val = float(data['phosphorus'])
    k_val = float(data['potassium'])
    crop_name = data['crop'].capitalize()  # Ensure proper case: 'rice' -> 'Rice'
    
    # =========================================================================
    # STEP 4: AGRONOMIC FEATURE ENGINEERING
    # These features encode domain knowledge about optimal growing conditions
    # =========================================================================
    
    # 1. Base features
    base_features = {
        'Year': now.year,
        'Month': now.month,
        'Rainfall_mm': rainfall,
        'Temperature_C': temperature,
        'Fertilizer': 75.0,  # Default value
        'N': n_val,
        'P': p_val,
        'K': k_val,
    }
    
    # 2. Binary stress/condition indicators
    agronomic_features = {
        'is_low_rainfall': 1 if rainfall < 300 else 0,
        'is_high_rainfall': 1 if rainfall > 700 else 0,
        'is_low_temp': 1 if temperature < 18 else 0,
        'is_high_temp': 1 if temperature > 30 else 0,
        'is_extreme_temp': 1 if temperature > 35 else 0,
    }
    
    # 3. Interaction features
    interaction_features = {
        'rain_temp_interaction': rainfall * temperature,
        'rain_squared': rainfall ** 2,
        'temp_squared': temperature ** 2,
    }
    
    # 4. Nutrient features
    total_npk = n_val + p_val + k_val
    nutrient_features = {
        'total_npk': total_npk,
        'npk_balance': abs(n_val - p_val) + abs(n_val - k_val) + abs(p_val - k_val),
        'n_ratio': n_val / (total_npk + 1e-5),
        'is_low_npk': 1 if total_npk < 100 else 0,
    }
    
    # 5. Crop-specific "good zone" indicators (AGRONOMIC KNOWLEDGE)
    # These encode when conditions are optimal for each crop
    good_zone_features = {
        # TEA: moderate rain (500-900mm), cool-moderate temp (18-26°C)
        'good_tea_zone': 1 if (crop_name == 'Tea' and 500 <= rainfall <= 900 and 18 <= temperature <= 26) else 0,
        
        # RUBBER: high rain (600-1000mm), warm temp (20-28°C)
        'good_rubber_zone': 1 if (crop_name == 'Rubber' and 600 <= rainfall <= 1000 and 20 <= temperature <= 28) else 0,
        
        # SUGARCANE: very high rain (700-1000mm), warm temp (22-30°C)
        'good_sugarcane_zone': 1 if (crop_name == 'Sugarcane' and 700 <= rainfall <= 1000 and 22 <= temperature <= 30) else 0,
        
        # CINNAMON: high rain (600-1000mm), warm temp (24-32°C)
        'good_cinnamon_zone': 1 if (crop_name == 'Cinnamon' and 600 <= rainfall <= 1000 and 24 <= temperature <= 32) else 0,
        
        # RICE: high rain (600+mm), moderate-warm temp (22-30°C)
        'good_rice_zone': 1 if (crop_name == 'Rice' and rainfall >= 600 and 22 <= temperature <= 30) else 0,
    }
    
    # 6. Stress indicators
    stress_features = {
        'drought_stress': 1 if (rainfall < 300 and temperature > 30) else 0,
        'cold_stress': 1 if temperature < 15 else 0,
        'heat_stress': 1 if temperature > 35 else 0,
    }
    
    # Combine all features
    all_raw_features = {
        **base_features,
        **agronomic_features,
        **interaction_features,
        **nutrient_features,
        **good_zone_features,
        **stress_features,
    }
    
    # =========================================================================
    # Build feature vector matching training feature order
    # =========================================================================
    feat_dict = {name: 0.0 for name in feature_names}
    
    # Update with calculated features
    for k, v in all_raw_features.items():
        if k in feat_dict:
            feat_dict[k] = v
    
    # Set Crop dummy (one-hot encoding)
    crop_col = f"Crop_{crop_name}"
    if crop_col in feat_dict:
        feat_dict[crop_col] = 1.0
    
    # District dummies stay 0 (defaulting to base district - Anuradhapura)
    
    # =========================================================================
    # Scale features using saved scaler params
    # =========================================================================
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


def postprocess_prediction(raw_prediction, crop):
    """
    STEP 6: Post-processing to ensure realistic, non-negative predictions.
    
    1. Inverse log transform (model predicts log(y+1))
    2. Enforce non-negative
    3. Clip to crop-specific realistic range
    """
    # 1. Inverse log transform: exp(y) - 1, then convert t/ha to kg/ha
    predicted_yield_t = np.expm1(raw_prediction)
    predicted_yield_kg = predicted_yield_t * 1000
    
    # 2. Enforce non-negative (STEP 7 requirement)
    predicted_yield_kg = max(predicted_yield_kg, MIN_YIELD_KG)
    
    # 3. Clip to crop-specific realistic range
    crop_lower = crop.lower()
    if crop_lower in CROP_YIELD_BOUNDS:
        lower_bound, upper_bound = CROP_YIELD_BOUNDS[crop_lower]
        predicted_yield_kg = max(min(predicted_yield_kg, upper_bound), lower_bound)
    
    return predicted_yield_kg


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model": Config.MODEL_TYPE,
        "features_loaded": len(feature_names) if feature_names else 0,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not available"}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    # Validate inputs
    error_msg = validate_inputs(data)
    if error_msg:
        logger.warning(f"Validation error: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    logger.info(f"Received prediction request for crop: {data['crop']}")
    
    try:
        # Preprocess with agronomic feature engineering
        input_vector = preprocess_and_enc(data)
        
        # Make prediction (model predicts log(yield_t/ha + 1))
        raw_prediction = model.predict(input_vector)[0]
        
        # Post-process: inverse transform + non-negative + clipping
        predicted_yield = postprocess_prediction(raw_prediction, data['crop'])
        
        # Calculate confidence range (±10%)
        conf_min = predicted_yield * 0.9
        conf_max = predicted_yield * 1.1
        
        # Confidence percentage (heuristic based on whether we're in "good zone")
        crop_name = data['crop'].capitalize()
        rainfall = float(data['rainfall'])
        temperature = float(data['temperature'])
        
        # Check if in optimal zone for confidence
        in_good_zone = False
        if crop_name == 'Rice' and rainfall >= 600 and 22 <= temperature <= 30:
            in_good_zone = True
        elif crop_name == 'Tea' and 500 <= rainfall <= 900 and 18 <= temperature <= 26:
            in_good_zone = True
        elif crop_name == 'Rubber' and 600 <= rainfall <= 1000 and 20 <= temperature <= 28:
            in_good_zone = True
        elif crop_name == 'Sugarcane' and 700 <= rainfall <= 1000 and 22 <= temperature <= 30:
            in_good_zone = True
        elif crop_name == 'Cinnamon' and 600 <= rainfall <= 1000 and 24 <= temperature <= 32:
            in_good_zone = True
        
        confidence_percentage = 95 if in_good_zone else 85
        
        # Feature importance for this prediction (heuristic)
        importances = model.feature_importances_
        feature_map = {
            'Rainfall_mm': 'Rainfall Intensity',
            'Temperature_C': 'Ambient Temperature',
            'N': 'Nitrogen Content',
            'P': 'Phosphorus Content',
            'K': 'Potassium Content',
            'total_npk': 'Soil Nutrient Density',
            'Fertilizer': 'Fertilizer Application',
            'rain_temp_interaction': 'Climate Balance',
            'good_rice_zone': 'Optimal Rice Conditions',
            'good_tea_zone': 'Optimal Tea Conditions',
            'good_rubber_zone': 'Optimal Rubber Conditions',
            'good_sugarcane_zone': 'Optimal Sugarcane Conditions',
            'good_cinnamon_zone': 'Optimal Cinnamon Conditions',
        }
        
        feat_imp_list = []
        for name, imp in zip(feature_names, importances):
            if name.startswith('Crop_'):
                if f"Crop_{crop_name}" == name:
                    display_name = f"{crop_name} Profile"
                    feat_imp_list.append({"name": display_name, "importance": float(imp)})
            elif name.startswith('District_'):
                continue  # Skip district features
            elif name in feature_map and imp > 0.01:
                feat_imp_list.append({"name": feature_map[name], "importance": float(imp)})

        # Sort and take top 5
        feat_imp_list = sorted(feat_imp_list, key=lambda x: x['importance'], reverse=True)[:5]
        
        # Normalize for frontend display (0-100 scale)
        if feat_imp_list:
            total_imp = sum(f['importance'] for f in feat_imp_list)
            if total_imp > 0:
                for f in feat_imp_list:
                    f['importance'] = round((f['importance'] / total_imp) * 70 + 15, 1)
        
        # Generate explanation
        if in_good_zone:
            explanation = f"Conditions are optimal for {crop_name.lower()} cultivation. The model predicts strong yields based on favorable rainfall and temperature."
        else:
            explanation = f"Current conditions may not be ideal for {crop_name.lower()}. Consider adjusting irrigation or planting timing for better results."
        
        response_data = {
            "success": True,
            "predicted_yield": round(predicted_yield, 2),
            "unit": Config.UNIT,
            "confidence_range": {
                "min": round(conf_min, 2),
                "max": round(conf_max, 2)
            },
            "confidence_percentage": confidence_percentage,
            "top_features": feat_imp_list,
            "explanation": explanation
        }
        
        # Save to database (Supabase)
        if supabase_client:
            try:
                supabase_client.table('predictions').insert({
                    'rainfall': float(data['rainfall']),
                    'temperature': float(data['temperature']),
                    'nitrogen': float(data['nitrogen']),
                    'phosphorus': float(data['phosphorus']),
                    'potassium': float(data['potassium']),
                    'crop_type': data['crop'],
                    'predicted_yield': float(predicted_yield),
                    'confidence': confidence_percentage
                }).execute()
                logger.info("Prediction saved to Supabase")
            except Exception as e:
                logger.error(f"Database error: {e}")
                # Don't fail the request if DB save fails

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "An internal error occurred during prediction"}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Return model performance and historical data for charts."""
    try:
        analytics_path = os.path.join(os.path.dirname(__file__), 'analytics.json')
        if not os.path.exists(analytics_path):
            return jsonify({"error": "Analytics not yet generated"}), 404
        
        with open(analytics_path, 'r') as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Return last 10 predictions from Supabase."""
    if not supabase_client:
        return jsonify({"error": "Database not connected"}), 503
    
    try:
        response = supabase_client.table('predictions') \
            .select('*') \
            .order('created_at', desc=True) \
            .limit(10) \
            .execute()
        
        return jsonify(response.data), 200
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
