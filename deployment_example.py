import joblib
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

class MemeCoinPricePredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model = joblib.load(os.path.join(model_dir, 'ensemble_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
        
        config_path = os.path.join(model_dir, 'model_config.json')
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.feature_names = self.config['feature_names']

    def predict_price(self, features_dict):
        X = np.array([features_dict[feat] for feat in self.feature_names]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        return prediction

    def get_model_info(self):
        return {
            'model_type': self.config['model_type'],
            'models': self.config['models'],
            'weights': self.config['weights'],
            'test_r2': self.config['test_r2'],
            'test_rmse': self.config['test_rmse'],
            'n_features': self.config['n_features']
        }

if __name__ == "__main__":
    predictor = MemeCoinPricePredictor()
    print(predictor.get_model_info())
