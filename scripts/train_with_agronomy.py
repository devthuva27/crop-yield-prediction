"""
COMPREHENSIVE TRAINING SCRIPT WITH AGRONOMIC DOMAIN KNOWLEDGE
==============================================================
This script implements Steps 3-6 of the crop yield prediction fix:
- Step 3: Agronomic pattern encoding
- Step 4: Feature engineering with domain knowledge
- Step 5: Retrain model with improved features
- Step 6: Post-processing for realistic outputs
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

# Create directories
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

# ============================================================================
# STEP 3 & 4: AGRONOMIC FEATURE ENGINEERING
# ============================================================================

def add_agronomic_features(df, crop_col='Crop', rain_col='Rainfall_mm', temp_col='Temperature_C',
                           n_col='N', p_col='P', k_col='K'):
    """
    Add domain-knowledge features based on agronomic patterns.
    These features help the model understand crop-specific optimal conditions.
    """
    df = df.copy()
    
    # Basic feature transformations
    df['is_low_rainfall'] = (df[rain_col] < 300).astype(int)
    df['is_high_rainfall'] = (df[rain_col] > 700).astype(int)
    df['is_low_temp'] = (df[temp_col] < 18).astype(int)
    df['is_high_temp'] = (df[temp_col] > 30).astype(int)
    df['is_extreme_temp'] = (df[temp_col] > 35).astype(int)
    
    # Interactions
    df['rain_temp_interaction'] = df[rain_col] * df[temp_col]
    df['rain_squared'] = df[rain_col] ** 2
    df['temp_squared'] = df[temp_col] ** 2
    
    # Nutrient features
    df['total_npk'] = df[n_col] + df[p_col] + df[k_col]
    df['npk_balance'] = (abs(df[n_col] - df[p_col]) + 
                         abs(df[n_col] - df[k_col]) + 
                         abs(df[p_col] - df[k_col]))
    df['n_ratio'] = df[n_col] / (df['total_npk'] + 1e-5)
    df['is_low_npk'] = (df['total_npk'] < 100).astype(int)
    
    # Crop-specific "good zone" indicators based on AGRONOMIC KNOWLEDGE
    # These encode when conditions are optimal for each crop
    
    # TEA: likes moderate rain (500-900mm), cool-moderate temp (18-26°C)
    df['good_tea_zone'] = (
        (df[crop_col] == 'Tea') &
        (df[rain_col] >= 500) & (df[rain_col] <= 900) &
        (df[temp_col] >= 18) & (df[temp_col] <= 26)
    ).astype(int)
    
    # RUBBER: likes high rain (600-1000mm), warm temp (20-28°C)
    df['good_rubber_zone'] = (
        (df[crop_col] == 'Rubber') &
        (df[rain_col] >= 600) & (df[rain_col] <= 1000) &
        (df[temp_col] >= 20) & (df[temp_col] <= 28)
    ).astype(int)
    
    # SUGARCANE: likes very high rain (700-1000mm), warm temp (22-30°C)
    df['good_sugarcane_zone'] = (
        (df[crop_col] == 'Sugarcane') &
        (df[rain_col] >= 700) & (df[rain_col] <= 1000) &
        (df[temp_col] >= 22) & (df[temp_col] <= 30)
    ).astype(int)
    
    # CINNAMON: likes high rain (600-1000mm), warm temp (24-32°C)
    df['good_cinnamon_zone'] = (
        (df[crop_col] == 'Cinnamon') &
        (df[rain_col] >= 600) & (df[rain_col] <= 1000) &
        (df[temp_col] >= 24) & (df[temp_col] <= 32)
    ).astype(int)
    
    # RICE: likes high rain, moderate-warm temp (22-30°C)
    df['good_rice_zone'] = (
        (df[crop_col] == 'Rice') &
        (df[rain_col] >= 600) &
        (df[temp_col] >= 22) & (df[temp_col] <= 30)
    ).astype(int)
    
    # Stress indicators (negative conditions)
    df['drought_stress'] = ((df[rain_col] < 300) & (df[temp_col] > 30)).astype(int)
    df['cold_stress'] = (df[temp_col] < 15).astype(int)
    df['heat_stress'] = (df[temp_col] > 35).astype(int)
    
    return df


def prepare_training_data():
    """Load and prepare training data with agronomic features."""
    print("Loading raw data...")
    df = pd.read_csv('../data/crop_data.csv')
    
    print(f"Total rows: {len(df)}")
    print(f"Crops: {df['Crop'].unique()}")
    print(f"\nSamples per crop:")
    print(df['Crop'].value_counts())
    
    # Add agronomic features
    print("\nAdding agronomic domain features...")
    df = add_agronomic_features(df)
    
    # Separate target
    target_col = 'Yield_t_per_ha'
    y = df[target_col]
    
    # Drop non-feature columns
    drop_cols = [target_col, 'Country']  # Country is always 'Sri Lanka'
    X = df.drop(columns=drop_cols)
    
    # One-hot encode categorical columns  
    categorical_cols = ['Crop', 'District']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Ensure all columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    return X, y, df


def train_model_with_domain_knowledge():
    """Train XGBoost with domain-knowledge features and regularization."""
    print("="*60)
    print("TRAINING MODEL WITH AGRONOMIC DOMAIN KNOWLEDGE")
    print("="*60)
    
    X, y, original_df = prepare_training_data()
    
    print(f"\nFeature count: {len(X.columns)}")
    print(f"New agronomic features added:")
    agro_feats = [c for c in X.columns if any(x in c for x in 
                  ['good_', 'stress', 'is_low', 'is_high', 'is_extreme', 
                   'npk_balance', 'total_npk', 'rain_temp', 'squared'])]
    for f in agro_feats:
        print(f"   - {f}")
    
    # Stratified split by crop
    # We'll use the original crop column for stratification
    crop_for_strat = original_df['Crop']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=crop_for_strat
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Save feature names and scaler params
    feature_names = list(X.columns)
    scaler_params = {}
    for col in X.columns:
        scaler_params[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max())
        }
    
    # Scale features (MinMax)
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
    
    print("\nTraining XGBoost with regularization...")
    # XGBoost with increased regularization to prevent wild extrapolation
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,      # Increased to reduce overfitting
        reg_alpha=0.1,           # L1 regularization
        reg_lambda=1.0,          # L2 regularization
        gamma=0.1,               # Min loss reduction for split
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train_log)
    
    # Predictions (inverse log transform)
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)  # Non-negative
    
    # Global metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"\nGLOBAL METRICS:")
    print(f"   MAE: {mae:.4f} t/ha")
    print(f"   RMSE: {rmse:.4f} t/ha")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2%}")
    
    # Per-crop metrics
    print("\nPER-CROP METRICS:")
    test_df = X_test.copy()
    test_df['y_true'] = y_test.values
    test_df['y_pred'] = y_pred
    
    # Find crop columns
    crop_cols = [c for c in X_test.columns if c.startswith('Crop_')]
    for crop_col in crop_cols:
        crop_name = crop_col.replace('Crop_', '')
        mask = test_df[crop_col] == 1
        if mask.sum() > 0:
            crop_y_true = test_df.loc[mask, 'y_true']
            crop_y_pred = test_df.loc[mask, 'y_pred']
            crop_r2 = r2_score(crop_y_true, crop_y_pred)
            crop_mae = mean_absolute_error(crop_y_true, crop_y_pred)
            print(f"   {crop_name}: R²={crop_r2:.4f}, MAE={crop_mae:.4f} t/ha")
    
    # Save model and metadata
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
    
    # Visualizations
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield (t/ha)')
    plt.ylabel('Predicted Yield (t/ha)')
    plt.title('XGBoost with Agronomic Features: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('../results/xgboost_actual_vs_predicted.png')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    top_15 = importance_df.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_15)
    plt.title('Top 15 Feature Importances (with Agronomic Features)')
    plt.tight_layout()
    plt.savefig('../results/xgboost_feature_importance.png')
    plt.close()
    
    print("\nDone! Model saved to models/best_model.pkl")
    return model, feature_names, scaler_params


if __name__ == "__main__":
    train_model_with_domain_knowledge()
