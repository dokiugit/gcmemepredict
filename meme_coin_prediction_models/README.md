# Meme Coin Price Prediction Model

## Overview
This is an ensemble regression model for predicting DOGE (Dogecoin) prices.

### Model Architecture
- **Type**: Ensemble Voting Regressor
- **Base Models**:
  - Random Forest (weight: 0.4)
  - Gradient Boosting (weight: 0.4)
  - AdaBoost (weight: 0.2)

### Performance Metrics
- **Test R² Score**: 0.953240
- **Test RMSE**: $0.01222974
- **Test MAE**: $0.00840235
- **Test MAPE**: 4.8050%

### Files Included
1. `ensemble_model.pkl` - Main ensemble model
2. `random_forest_model.pkl` - Random Forest model
3. `gradient_boosting_model.pkl` - Gradient Boosting model
4. `adaboost_model.pkl` - AdaBoost model
5. `feature_scaler.pkl` - Feature scaling transformer
6. `target_scaler.pkl` - Target scaling transformer
7. `model_config.json` - Model configuration and metadata
8. `feature_importance_rf.csv` - RF feature importance
9. `feature_importance_gb.csv` - GB feature importance
10. `model_comparison.csv` - Comparison metrics
11. `deployment_example.py` - Example code for production use

### Features Used (28)
day_of_week, day_of_month, month, quarter, is_weekend, price_change, price_pct_change, price_volatility_7d, price_volatility_14d, sma_7...

### Training Data
- **Training Samples**: 268
- **Testing Samples**: 68
- **Historical Data**: 364 days

### Usage
```python
import joblib

# Load model
model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Prepare features and make prediction
X_scaled = scaler.transform(features)
prediction = model.predict(X_scaled)
```

### Data Source
- **API**: CoinGecko (Free, Real-time)
- **Endpoint**: https://api.coingecko.com/api/v3/coins/

### Prediction Target
- **Horizon**: Next day price
- **Coin**: DOGE (Dogecoin)

### Model Retraining
The model should be retrained monthly with updated data to maintain accuracy.
