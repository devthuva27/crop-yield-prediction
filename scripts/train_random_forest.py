
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os

# Create directories if they don't exist
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

def train_random_forest():
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

    print("Training Random Forest model...")
    # 3. Train Random Forest with specified parameters
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)

    print("Making predictions...")
    # 4. Make predictions on test data
    y_pred = rf_model.predict(X_test)

    # 5. Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.4f}")

    # 6. Get feature importance
    print("Extracting feature importance...")
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    # Save top 15 features
    top_15_features = feature_importance.head(15)
    top_15_features.to_csv('../results/random_forest_feature_importance.csv', index=False)

    # 7. Visualizations
    print("Creating visualizations...")
    
    # Plot: Actual vs Predicted Yield (scatter)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Random Forest: Actual vs Predicted Yield')
    plt.tight_layout()
    plt.savefig('../results/rf_actual_vs_predicted.png')
    plt.close()

    # Plot: Feature Importance (bar chart) - Top 15
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_15_features)
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('../results/rf_feature_importance.png')
    plt.close()

    # 8. Save model
    print("Saving model...")
    joblib.dump(rf_model, '../models/random_forest_model.pkl')

    # 9. Save performance report
    print("Saving performance report...")
    report_content = f"""Random Forest Performance Report
================================
n_estimators: 100
max_depth: 15
random_state: 42

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

    with open('../results/random_forest_performance.txt', 'w') as f:
        f.write(report_content)

    print("Done!")

if __name__ == "__main__":
    train_random_forest()
