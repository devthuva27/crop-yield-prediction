
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os

# Create directories if they don't exist
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

def train_xgboost():
    print("Loading data...")
    # 1. Load training and test data
    train_df = pd.read_csv('../data/processed_train.csv')
    test_df = pd.read_csv('../data/processed_test.csv')

    # 2. Prepare data
    # X (features) = all columns except Yield
    # y (target) = Yield column
    target_col = 'Yield_t_per_ha'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print("Training XGBoost model (with log-transformed target)...")
    # 3. Train XGBoost with specified parameters
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    # Use log transformation on target
    y_train_log = np.log1p(y_train)
    xgb_model.fit(X_train, y_train_log)

    print("Making predictions...")
    # 4. Make predictions on test data (in log scale)
    y_pred_log = xgb_model.predict(X_test)
    # Inverse transform to get back to original scale
    y_pred = np.expm1(y_pred_log)
    
    # Ensure non-negative
    y_pred = np.maximum(y_pred, 0)

    # 5. Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.4f}")

    # 6. Get feature importance
    print("Extracting feature importance...")
    # XGBoost provides different types of importance, "weight", "gain", "cover". 
    # Default feature_importances_ property is usually gain or similar.
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': xgb_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    # Save top 15 features
    top_15_features = feature_importance.head(15)
    top_15_features.to_csv('../results/xgboost_feature_importance.csv', index=False)

    # 7. Visualizations
    print("Creating visualizations...")
    
    # Plot: Actual vs Predicted (scatter)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('XGBoost: Actual vs Predicted Yield')
    plt.tight_layout()
    plt.savefig('../results/xgboost_actual_vs_predicted.png')
    plt.close()

    # Plot: Feature Importance (bar chart) - Top 15
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_15_features)
    plt.title('Top 15 Feature Importances (XGBoost)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('../results/xgboost_feature_importance.png')
    plt.close()

    # Plot: Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Yield')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('XGBoost: Residual Plot')
    plt.tight_layout()
    plt.savefig('../results/xgboost_residuals.png')
    plt.close()

    # 8. Save model
    print("Saving model...")
    joblib.dump(xgb_model, '../models/xgboost_model.pkl')

    # 9. Save performance report
    print("Saving performance report...")
    
    # Hardcoded values from previous steps (verified via tools)
    lr_r2 = 0.9953 
    rf_r2 = 0.9948
    
    comparison_text = f"""Model Comparison (R2 Score):
----------------------------
Linear Regression: {lr_r2:.4f}
Random Forest:     {rf_r2:.4f}
XGBoost:           {r2:.4f}

Conclusion:
"""
    if r2 > max(lr_r2, rf_r2):
        comparison_text += "XGBoost performs the best among the three models."
    elif r2 > lr_r2:  
        comparison_text += "XGBoost performs better than Linear Regression but worse than Random Forest (if RF was higher)." # logic check
    elif r2 < min(lr_r2, rf_r2):
        comparison_text += "XGBoost performs slightly worse than Linear Regression and Random Forest, possibly due to overfitting or need for more tuning, although all models are very high performing."
    else:
        comparison_text += "XGBoost performance is comparable to the other models."

    report_content = f"""XGBoost Performance Report
================================
n_estimators: 100
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
random_state: 42
objective: reg:squarederror

Metrics:
--------
Mean Absolute Error (MAE): {mae:.4f}
Root Mean Squared Error (RMSE): {rmse:.4f}
R² Score: {r2:.4f}
Mean Absolute Percentage Error (MAPE): {mape:.4%}

Top 15 Feature Importances:
---------------------------
"""
    for index, row in top_15_features.iterrows():
        report_content += f"{row['Feature']}: {row['Importance']:.6f}\n"

    report_content += "\n" + comparison_text

    with open('../results/xgboost_performance.txt', 'w') as f:
        f.write(report_content)

    print("Done!")

if __name__ == "__main__":
    train_xgboost()
