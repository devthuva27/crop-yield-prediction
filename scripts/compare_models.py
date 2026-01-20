
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import shutil

# Ensure directories
os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# Define NumpyLinearRegression class for unpickling compatibility
# This handles the case where the model was trained using the fallback class
class NumpyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X_b = np.c_[np.ones((len(X), 1)), X]
        theta_best = np.linalg.pinv(X_b).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]
        
    def predict(self, X):
        X = np.array(X)
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(np.r_[self.intercept_, self.coef_])

def compare_models():
    print("Loading data...")
    test_df = pd.read_csv('../data/processed_test.csv')
    
    target_col = 'Yield_t_per_ha'
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print("Loading models...")
    models = {}
    
    # Load Linear Regression (Standard Pickle)
    try:
        with open('../models/linear_regression_model.pkl', 'rb') as f:
            models['Linear Regression'] = pickle.load(f)
    except Exception as e:
        print(f"Error loading Linear Regression: {e}")

    # Load Random Forest (Joblib)
    try:
        models['Random Forest'] = joblib.load('../models/random_forest_model.pkl')
    except Exception as e:
        print(f"Error loading Random Forest: {e}")

    # Load XGBoost (Joblib)
    try:
        models['XGBoost'] = joblib.load('../models/xgboost_model.pkl')
    except Exception as e:
        print(f"Error loading XGBoost: {e}")

    results = []

    print("Evaluating models...")
    for name, model in models.items():
        try:
            print(f"Predicting with {name}...")
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Safe MAPE
            epsilon = 1e-10
            # Handle potential pandas series vs numpy array issues
            y_true_safe = np.array(y_test)
            y_true_safe[y_true_safe == 0] = epsilon
            mape = np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100

            results.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R2_Score': r2,
                'MAPE': mape
            })
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Rank by R2 Score (descending)
    results_df = results_df.sort_values(by='R2_Score', ascending=False).reset_index(drop=True)
    results_df['Rank'] = results_df.index + 1
    
    # Add Interpretation
    def interpret(row):
        if row['Rank'] == 1:
            return "BEST! Highest accuracy"
        elif row['Rank'] == 2:
            return "Good but slightly lower accuracy"
        else:
            return "Performance is lower relative to others"

    results_df['Interpretation'] = results_df.apply(interpret, axis=1)
    
    # Format MAPE for display (csv saving usually prefers raw numbers, but user asked for example with %)
    # We will save raw numbers for calculation but maybe format string for final CSV output if strictly requested.
    # User format example: "7.24%". 
    results_df_display = results_df.copy()
    results_df_display['MAPE'] = results_df_display['MAPE'].apply(lambda x: f"{x:.2f}%")
    results_df_display['MAE'] = results_df_display['MAE'].round(2)
    results_df_display['RMSE'] = results_df_display['RMSE'].round(2)
    results_df_display['R2_Score'] = results_df_display['R2_Score'].round(4)
    
    print("Saving comparison CSV...")
    results_df_display.to_csv('../results/model_comparison.csv', index=False)
    print(results_df_display)

    # 5. Visualizations
    print("Creating visualizations...")
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # R2 Score Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='R2_Score', data=results_df, palette='viridis')
    plt.title('Model Comparison: R² Score')
    plt.ylabel('R² Score')
    plt.ylim(0, 1.1) # Assuming 0-1 range
    for i, v in enumerate(results_df['R2_Score']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('../results/model_comparison_r2.png')
    plt.close()

    # MAE Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='MAE', data=results_df, palette='magma')
    plt.title('Model Comparison: Mean Absolute Error (MAE)')
    plt.ylabel('MAE (t/ha)')
    for i, v in enumerate(results_df['MAE']):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('../results/model_comparison_mae.png')
    plt.close()
    
    # MAPE Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='MAPE', data=results_df, palette='rocket')
    plt.title('Model Comparison: Mean Absolute Percentage Error (MAPE)')
    plt.ylabel('MAPE (%)')
    for i, v in enumerate(results_df['MAPE']):
        plt.text(i, v, f"{v:.2f}%", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('../results/model_comparison_mape.png')
    plt.close()

    # 6. Summary File & Save Best Model
    best_model_name = results_df.iloc[0]['Model']
    best_r2 = results_df.iloc[0]['R2_Score']
    worst_r2 = results_df.iloc[-1]['R2_Score']
    
    improvement = ((best_r2 - worst_r2) / worst_r2) * 100 if worst_r2 != 0 else 0
    
    summary_content = f"""Model Comparison Summary
========================
Best Model: {best_model_name}

Why it wins: 
It achieved the highest R² Score ({best_r2:.4f}) and superior error metrics compared to other models.

Top Metrics ({best_model_name}):
- MAE: {results_df.iloc[0]['MAE']:.4f}
- RMSE: {results_df.iloc[0]['RMSE']:.4f}
- R²: {best_r2:.4f}
- MAPE: {results_df.iloc[0]['MAPE']:.2f}%

Rankings:
"""
    for index, row in results_df.iterrows():
        summary_content += f"{row['Rank']}. {row['Model']} (R²: {row['R2_Score']:.4f})\n"

    summary_content += f"\nConclusion:\nThe {best_model_name} provides the most accurate yield predictions."
    
    with open('../results/model_comparison_summary.txt', 'w') as f:
        f.write(summary_content)
        
    print(f"Comparison summary saved. Best model identified as: {best_model_name}")

    # Save best model to best_model.pkl
    # Note: User requested "copy of XGBoost" specifically in prompt example, but we act on logic.
    # But if the user explicitly wants XGBoost regardless of score, we might deviate. 
    # Current decision: Save the ACTUAL best model.
    
    if best_model_name in models:
        best_model_obj = models[best_model_name]
        try:
             joblib.dump(best_model_obj, '../models/best_model.pkl')
             print(f"Saved {best_model_name} as models/best_model.pkl")
        except Exception as e:
            print(f"Error saving best model: {e}")

if __name__ == "__main__":
    compare_models()
