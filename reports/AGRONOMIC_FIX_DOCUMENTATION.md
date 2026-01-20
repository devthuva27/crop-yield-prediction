# Crop Yield Prediction Model - Agronomic Enhancement

## Problem Statement
The original model was producing **negative or unrealistic predictions** for tea, rubber, sugarcane, and cinnamon crops, even with reasonable inputs. Only rice predictions were acceptable.

## Root Cause Analysis
1. **Data Issue**: The original training data lacked strong correlations between environmental conditions (rainfall, temperature) and yield outcomes
2. **No Domain Knowledge**: The model couldn't learn agronomic patterns that weren't present in the data
3. **Scale Imbalance**: Yield ranges varied dramatically across crops (0.5-88 t/ha)

## Solution Implemented

### 1. Agronomic Feature Engineering
Added domain-knowledge features to help the model understand optimal growing conditions:

```python
# Binary stress/condition indicators
is_low_rainfall = rainfall < 300
is_high_rainfall = rainfall > 700
is_low_temp = temperature < 18
is_high_temp = temperature > 30
is_extreme_temp = temperature > 35

# Crop-specific optimal zone indicators
good_rice_zone = (crop == 'Rice' and rainfall >= 600 and 22 <= temp <= 30)
good_tea_zone = (crop == 'Tea' and 500 <= rainfall <= 900 and 18 <= temp <= 26)
good_rubber_zone = (crop == 'Rubber' and 600 <= rainfall <= 1000 and 20 <= temp <= 28)
good_sugarcane_zone = (crop == 'Sugarcane' and 700 <= rainfall <= 1000 and 22 <= temp <= 30)
good_cinnamon_zone = (crop == 'Cinnamon' and 600 <= rainfall <= 1000 and 24 <= temp <= 32)

# Stress indicators
drought_stress = (rainfall < 300 and temperature > 30)
heat_stress = (temperature > 35)
cold_stress = (temperature < 15)
```

### 2. Synthetic Data Augmentation
Generated synthetic training examples that encode agronomic knowledge:
- ~700 examples per crop (500 random + 100 good + 100 bad conditions)
- Good conditions → high yields (95-115% of base)
- Bad conditions → low yields (25-45% of base)

### 3. Regularized XGBoost Model
```python
XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=5,      # Prevent overfitting
    reg_alpha=0.5,           # L1 regularization
    reg_lambda=2.0,          # L2 regularization
    gamma=0.2,               # Min split improvement
)
```

### 4. Post-Processing Safety Layer
```python
# 1. Inverse log transform
predicted_yield_t = np.expm1(raw_prediction)

# 2. Non-negative constraint
predicted_yield_kg = max(predicted_yield_t * 1000, 0.0)

# 3. Crop-specific realistic range clipping
CROP_YIELD_BOUNDS = {
    'rice': (0, 15000),        # kg/ha
    'tea': (0, 5000),
    'rubber': (0, 4000),
    'sugarcane': (0, 150000),
    'cinnamon': (0, 3000),
}
```

## Validation Results

### Monotonicity Tests (Good > Bad Conditions)
| Crop | Bad Scenario | Good Scenario | Status |
|------|-------------|---------------|--------|
| Rice | 2,565 kg/ha | 4,488 kg/ha | ✓ PASS |
| Tea | 1,265 kg/ha | 2,265 kg/ha | ✓ PASS |
| Rubber | 1,377 kg/ha | 1,587 kg/ha | ✓ PASS |
| Sugarcane | 26,896 kg/ha | 79,196 kg/ha | ✓ PASS |
| Cinnamon | 566 kg/ha | 667 kg/ha | ✓ PASS |

### Key Improvements
- ✅ All predictions are non-negative
- ✅ Good conditions produce higher yields than bad conditions
- ✅ Predictions are within realistic crop-specific ranges
- ✅ Model respects agronomic domain knowledge

## Files Modified

### Training
- `train_with_augmentation.py` - New training script with data augmentation

### Backend
- `backend/app.py` - Updated with agronomic feature engineering and post-processing

### Model Artifacts
- `models/best_model.pkl` - Retrained model
- `models/feature_scaler.json` - Updated scaler with new features
- `feature_names.txt` - Updated feature list

## API Behavior
The `/predict` endpoint now:
1. Validates inputs (rainfall 100-1000mm, temp 10-40°C, etc.)
2. Computes agronomic features (good zones, stress indicators)
3. Scales features using saved parameters
4. Makes prediction with log-scale model
5. Post-processes: inverse log → non-negative → crop-specific clipping
6. Returns realistic, positive yield in kg/ha

## Agronomic Knowledge Encoded

### Optimal Conditions by Crop
| Crop | Rainfall (mm) | Temperature (°C) |
|------|---------------|------------------|
| Rice | 600-1000 | 22-30 |
| Tea | 500-900 | 18-26 |
| Rubber | 600-1000 | 20-28 |
| Sugarcane | 700-1000 | 22-30 |
| Cinnamon | 600-1000 | 24-32 |

### Stress Conditions (Yield Reduction)
- Drought: Rainfall < 300mm → 30-50% yield reduction
- Heat Stress: Temp > 35°C → Significant yield loss
- Cold Stress: Temp < 15°C → Growth inhibition
- Nutrient Deficiency: Total NPK < 100 → Reduced yields
