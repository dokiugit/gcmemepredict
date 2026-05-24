
# Production Deployment Code
# ===========================

import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime

class MemeCoinPricePredictor:
    def __init__(self, model_dir='meme_coin_prediction_models'):
        """Initialize the predictor with saved models"""
        # Load model
        self.model = joblib.load(f'{model_dir}/ensemble_model.pkl')
        
        # Load scalers
        self.scaler = joblib.load(f'{model_dir}/feature_scaler.pkl')
        
        # Load configuration
        with open(f'{model_dir}/model_config.json') as f:
            self.config = json.load(f)
        
        self.feature_names = self.config['feature_names']
    
    def predict_price(self, features_dict):
        """Predict next day price given current features"""
        # Create feature array
        X = np.array([features_dict[feat] for feat in self.feature_names]).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': self.config['model_type'],
            'models': self.config['models'],
            'weights': self.config['weights'],
            'test_r2': self.config['test_r2'],
            'test_rmse': self.config['test_rmse'],
            'n_features': self.config['n_features']
        }

# Usage Example
if __name__ == "__main__":
    # Initialize predictor
    predictor = MemeCoinPricePredictor()
    
    # Get model info
    print(predictor.get_model_info())
    
    # Make prediction (provide features)
    # features = {
    #     'price_change': 0.001,
    #     'volume_change': 1000000,
    #     # ... add all other features
    # }
    # prediction = predictor.predict_price(features)
    # print(f"Predicted price: ${prediction:.8f}")
