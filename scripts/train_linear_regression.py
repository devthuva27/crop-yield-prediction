
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Paths
train_path = '../data/processed_train.csv'
test_path = '../data/processed_test.csv'
model_path = '../models/linear_regression_model.pkl'
plot_path = '../output/plots/linear_regression_predictions.png'
report_path = '../results/linear_regression_performance.txt'

# Ensure directories
os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)
os.makedirs('../output/plots', exist_ok=True)

# --- 1. Load Data ---
print("Loading data...")
if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("Error: Data files not found.")
    exit(1)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

target_col = 'Yield_t_per_ha'

# Split X and y
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]
X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col]

print(f"Training Data: {X_train.shape}")
print(f"Test Data:     {X_test.shape}")

# --- 2. Train Model (Robust Import) ---
model = None
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    print("Using scikit-learn LinearRegression.")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

except ImportError:
    print("WARNING: scikit-learn not found. Using NumPy implementation (OLS).")
    
    class NumpyLinearRegression:
        def __init__(self):
            self.coef_ = None
            self.intercept_ = None
            
        def fit(self, X, y):
            # Add intercept term (column of ones)
            X = np.array(X)
            y = np.array(y)
            X_b = np.c_[np.ones((len(X), 1)), X]
            
            # Normal Equation: theta = (X.T * X)^-1 * X.T * y
            # Using pseudoinverse for stability: pinv(X) * y
            theta_best = np.linalg.pinv(X_b).dot(y)
            
            self.intercept_ = theta_best[0]
            self.coef_ = theta_best[1:]
            
        def predict(self, X):
            X = np.array(X)
            X_b = np.c_[np.ones((len(X), 1)), X]
            return X_b.dot(np.r_[self.intercept_, self.coef_])
            
    model = NumpyLinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Manual Metrics
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_test - y_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2 = 1 - (ss_res / ss_tot)

# Calculate MAPE (Handle div by zero)
# Avoid division by zero by replacing 0 with a small epsilon
epsilon = 1e-10
y_test_safe = y_test.replace(0, epsilon)
mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100

print("Training complete.")

# --- 3. Save Model ---
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

# --- 4. Visualizations ---
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')

# Diagonal line
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plt.title('Actual vs Predicted Yield (Linear Regression)')
plt.xlabel('Actual Yield (t/ha)')
plt.ylabel('Predicted Yield (t/ha)')
plt.grid(True)
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# --- 5. Report ---
report_content = [
    "--- Linear Regression Performance Report ---",
    f"Model Type: {'scikit-learn LinearRegression' if 'sklearn' in str(type(model)) else 'NumPy OLS (Normal Equation)'}",
    "",
    "Performance Metrics:",
    f"MAE:  {mae:.4f} t/ha",
    f"MAE (kg): {mae*1000:.2f} kg/ha",
    f"RMSE: {rmse:.4f} t/ha",
    f"R2 Score: {r2:.4f}",
    f"MAPE: {mape:.2f}%",
    "",
    "Conclusion:"
]

if r2 > 0.85:
    report_content.append("Model Performance: EXCELLENT. The model explains a very high portion of variance.")
elif r2 > 0.70:
    report_content.append("Model Performance: GOOD. Useful for general predictions.")
else:
    report_content.append("Model Performance: POOR. Consider non-linear models (Random Forest, XGBoost).")

with open(report_path, 'w') as f:
    f.write("\n".join(report_content))

print(f"Report saved to {report_path}")
print("\nMetrics:")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
