"""
COMPREHENSIVE TRAINING WITH AGRONOMIC DATA AUGMENTATION
========================================================
The issue: The original data doesn't have strong correlations between 
environmental conditions and yield. The model can't learn agronomic 
patterns that don't exist in the data.

SOLUTION: Augment training data with synthetic examples that encode
agronomic knowledge, giving the model examples of good vs bad conditions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib
import json
import os

os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

# ============================================================================
# AGRONOMIC KNOWLEDGE BASE
# ============================================================================
CROP_PARAMS = {
    'Rice': {
        'base_yield': 4.5,  # t/ha
        'optimal_rain': (600, 1000),
        'optimal_temp': (22, 30),
        'rain_sensitivity': 0.8,  # How much yield drops with bad rain
        'temp_sensitivity': 0.7,
    },
    'Tea': {
        'base_yield': 2.3,
        'optimal_rain': (500, 900),
        'optimal_temp': (18, 26),
        'rain_sensitivity': 0.9,
        'temp_sensitivity': 0.8,
    },
    'Rubber': {
        'base_yield': 1.5,
        'optimal_rain': (600, 1000),
        'optimal_temp': (20, 28),
        'rain_sensitivity': 0.85,
        'temp_sensitivity': 0.75,
    },
    'Sugarcane': {
        'base_yield': 75,
        'optimal_rain': (700, 1000),
        'optimal_temp': (22, 30),
        'rain_sensitivity': 0.7,
        'temp_sensitivity': 0.6,
    },
    'Cinnamon': {
        'base_yield': 0.65,
        'optimal_rain': (600, 1000),
        'optimal_temp': (24, 32),
        'rain_sensitivity': 0.8,
        'temp_sensitivity': 0.7,
    },
}


def calculate_agronomic_yield_factor(crop, rainfall, temperature, n, p, k):
    """
    Calculate a yield adjustment factor based on agronomic conditions.
    Returns a multiplier between 0.3 and 1.2.
    """
    if crop not in CROP_PARAMS:
        return 1.0
    
    params = CROP_PARAMS[crop]
    rain_opt = params['optimal_rain']
    temp_opt = params['optimal_temp']
    
    # Rainfall factor
    if rain_opt[0] <= rainfall <= rain_opt[1]:
        rain_factor = 1.0
    elif rainfall < rain_opt[0]:
        deficit = (rain_opt[0] - rainfall) / rain_opt[0]
        rain_factor = max(0.3, 1.0 - deficit * params['rain_sensitivity'])
    else:  # rainfall > rain_opt[1]
        excess = (rainfall - rain_opt[1]) / rain_opt[1]
        rain_factor = max(0.6, 1.0 - excess * 0.3)  # Slight penalty for excess
    
    # Temperature factor
    if temp_opt[0] <= temperature <= temp_opt[1]:
        temp_factor = 1.0
    elif temperature < temp_opt[0]:
        deficit = (temp_opt[0] - temperature) / temp_opt[0]
        temp_factor = max(0.4, 1.0 - deficit * params['temp_sensitivity'])
    else:  # temperature > temp_opt[1]
        excess = (temperature - temp_opt[1]) / 10  # Scale by 10°C
        temp_factor = max(0.3, 1.0 - excess * params['temp_sensitivity'])
    
    # Nutrient factor
    total_npk = n + p + k
    if total_npk >= 150:
        npk_factor = 1.1
    elif total_npk >= 100:
        npk_factor = 1.0
    elif total_npk >= 50:
        npk_factor = 0.85
    else:
        npk_factor = 0.7
    
    # Combined factor
    combined = rain_factor * temp_factor * npk_factor
    return np.clip(combined, 0.3, 1.2)


def add_agronomic_features(df, crop_col='Crop', rain_col='Rainfall_mm', temp_col='Temperature_C',
                           n_col='N', p_col='P', k_col='K'):
    """Add domain-knowledge features."""
    df = df.copy()
    
    # Basic indicators
    df['is_low_rainfall'] = (df[rain_col] < 300).astype(int)
    df['is_high_rainfall'] = (df[rain_col] > 700).astype(int)
    df['is_low_temp'] = (df[temp_col] < 18).astype(int)
    df['is_high_temp'] = (df[temp_col] > 30).astype(int)
    df['is_extreme_temp'] = (df[temp_col] > 35).astype(int)
    
    # Interactions
    df['rain_temp_interaction'] = df[rain_col] * df[temp_col]
    df['rain_squared'] = df[rain_col] ** 2
    df['temp_squared'] = df[temp_col] ** 2
    
    # Nutrients
    df['total_npk'] = df[n_col] + df[p_col] + df[k_col]
    df['npk_balance'] = (abs(df[n_col] - df[p_col]) + 
                         abs(df[n_col] - df[k_col]) + 
                         abs(df[p_col] - df[k_col]))
    df['n_ratio'] = df[n_col] / (df['total_npk'] + 1e-5)
    df['is_low_npk'] = (df['total_npk'] < 100).astype(int)
    
    # Crop-specific optimal zones
    df['good_tea_zone'] = ((df[crop_col] == 'Tea') &
        (df[rain_col] >= 500) & (df[rain_col] <= 900) &
        (df[temp_col] >= 18) & (df[temp_col] <= 26)).astype(int)
    
    df['good_rubber_zone'] = ((df[crop_col] == 'Rubber') &
        (df[rain_col] >= 600) & (df[rain_col] <= 1000) &
        (df[temp_col] >= 20) & (df[temp_col] <= 28)).astype(int)
    
    df['good_sugarcane_zone'] = ((df[crop_col] == 'Sugarcane') &
        (df[rain_col] >= 700) & (df[rain_col] <= 1000) &
        (df[temp_col] >= 22) & (df[temp_col] <= 30)).astype(int)
    
    df['good_cinnamon_zone'] = ((df[crop_col] == 'Cinnamon') &
        (df[rain_col] >= 600) & (df[rain_col] <= 1000) &
        (df[temp_col] >= 24) & (df[temp_col] <= 32)).astype(int)
    
    df['good_rice_zone'] = ((df[crop_col] == 'Rice') &
        (df[rain_col] >= 600) &
        (df[temp_col] >= 22) & (df[temp_col] <= 30)).astype(int)
    
    # Stress indicators
    df['drought_stress'] = ((df[rain_col] < 300) & (df[temp_col] > 30)).astype(int)
    df['cold_stress'] = (df[temp_col] < 15).astype(int)
    df['heat_stress'] = (df[temp_col] > 35).astype(int)
    
    return df


def generate_synthetic_examples(n_per_crop=500):
    """
    Generate synthetic training examples that encode agronomic knowledge.
    This teaches the model the relationship between conditions and yield.
    """
    synthetic_rows = []
    
    for crop, params in CROP_PARAMS.items():
        base_yield = params['base_yield']
        
        # Generate more examples for better coverage
        for _ in range(n_per_crop):
            # Random conditions across the input range
            rainfall = np.random.uniform(100, 1000)
            temperature = np.random.uniform(10, 40)
            n = np.random.uniform(0, 200)
            p = np.random.uniform(0, 100)
            k = np.random.uniform(0, 100)
            
            # Calculate agronomically-adjusted yield
            factor = calculate_agronomic_yield_factor(crop, rainfall, temperature, n, p, k)
            adjusted_yield = base_yield * factor * np.random.uniform(0.9, 1.1)
            adjusted_yield = max(adjusted_yield, 0.1)
            
            synthetic_rows.append({
                'Crop': crop,
                'Country': 'Sri Lanka',
                'District': 'Anuradhapura',
                'Year': 2024,
                'Month': np.random.randint(1, 13),
                'Rainfall_mm': rainfall,
                'Temperature_C': temperature,
                'Fertilizer': 75.0,
                'N': n,
                'P': p,
                'K': k,
                'Yield_t_per_ha': adjusted_yield
            })
        
        # Add explicit good/bad examples for each crop to reinforce patterns
        # GOOD conditions
        for _ in range(100):
            rain_opt = params['optimal_rain']
            temp_opt = params['optimal_temp']
            rainfall = np.random.uniform(rain_opt[0], rain_opt[1])
            temperature = np.random.uniform(temp_opt[0], temp_opt[1])
            n, p, k = np.random.uniform(80, 150), np.random.uniform(40, 80), np.random.uniform(40, 80)
            adjusted_yield = base_yield * np.random.uniform(0.95, 1.15)
            synthetic_rows.append({
                'Crop': crop, 'Country': 'Sri Lanka', 'District': 'Anuradhapura',
                'Year': 2024, 'Month': np.random.randint(1, 13),
                'Rainfall_mm': rainfall, 'Temperature_C': temperature,
                'Fertilizer': 75.0, 'N': n, 'P': p, 'K': k,
                'Yield_t_per_ha': adjusted_yield
            })
        
        # BAD conditions (drought + heat)
        for _ in range(100):
            rainfall = np.random.uniform(100, 250)
            temperature = np.random.uniform(33, 40)
            n, p, k = np.random.uniform(20, 80), np.random.uniform(10, 40), np.random.uniform(10, 40)
            adjusted_yield = base_yield * np.random.uniform(0.25, 0.45)
            synthetic_rows.append({
                'Crop': crop, 'Country': 'Sri Lanka', 'District': 'Anuradhapura',
                'Year': 2024, 'Month': np.random.randint(1, 13),
                'Rainfall_mm': rainfall, 'Temperature_C': temperature,
                'Fertilizer': 75.0, 'N': n, 'P': p, 'K': k,
                'Yield_t_per_ha': adjusted_yield
            })
    
    return pd.DataFrame(synthetic_rows)



def train_model_with_augmentation():
    """Train model with both real and synthetic data."""
    print("="*60)
    print("TRAINING WITH AGRONOMIC DATA AUGMENTATION")
    print("="*60)
    
    # Load real data
    print("Loading real data...")
    real_df = pd.read_csv('../data/crop_data.csv')
    print(f"Real data: {len(real_df)} rows")
    
    # Generate synthetic data
    print("Generating synthetic agronomic examples...")
    synthetic_df = generate_synthetic_examples(n_per_crop=300)
    print(f"Synthetic data: {len(synthetic_df)} rows")
    
    # Combine
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    print(f"Combined data: {len(combined_df)} rows")
    
    # Add agronomic features
    print("Adding agronomic features...")
    combined_df = add_agronomic_features(combined_df)
    
    # Prepare X and y
    target_col = 'Yield_t_per_ha'
    y = combined_df[target_col]
    
    drop_cols = [target_col, 'Country']
    X = combined_df.drop(columns=drop_cols)
    
    # One-hot encode
    categorical_cols = ['Crop', 'District']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Ensure numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    feature_names = list(X.columns)
    print(f"Total features: {len(feature_names)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save scaler params
    scaler_params = {}
    for col in X.columns:
        scaler_params[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max())
        }
    
    # Scale
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    for col in X.columns:
        min_val, max_val = scaler_params[col]['min'], scaler_params[col]['max']
        if max_val - min_val > 0:
            X_train_scaled[col] = (X_train[col] - min_val) / (max_val - min_val)
            X_test_scaled[col] = (X_test[col] - min_val) / (max_val - min_val)
        else:
            X_train_scaled[col] = 0.0
            X_test_scaled[col] = 0.0
    
    # Log transform target
    y_train_log = np.log1p(y_train)
    
    # Train
    print("\nTraining XGBoost with regularization...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.2,
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train_log)
    
    # Predict
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"\nGLOBAL METRICS:")
    print(f"   MAE: {mae:.4f} t/ha")
    print(f"   RMSE: {rmse:.4f} t/ha")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2%}")
    
    # Save
    print("\nSaving model and metadata...")
    joblib.dump(model, '../models/xgboost_model.pkl')
    joblib.dump(model, '../models/best_model.pkl')
    
    with open('../models/feature_scaler.json', 'w') as f:
        json.dump(scaler_params, f, indent=4)
    
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTOP 15 FEATURE IMPORTANCES:")
    for _, row in importance_df.head(15).iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")
    
    importance_df.to_csv('../results/xgboost_feature_importance.csv', index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield (t/ha)')
    plt.ylabel('Predicted Yield (t/ha)')
    plt.title('Agronomic-Augmented Model: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('../results/xgboost_actual_vs_predicted.png')
    plt.close()
    
    print("\nDone! Model saved to models/best_model.pkl")
    return model


if __name__ == "__main__":
    train_model_with_augmentation()
