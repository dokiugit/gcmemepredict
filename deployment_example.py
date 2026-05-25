"""
Meme Coin Price Prediction - Production Deployment Example
============================================================

This script demonstrates how to load and use the saved ensemble model
for predicting DOGE prices in a production environment.

Usage:
    python deployment_example.py
    
Author: ML Team
Date: 2026
"""

import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Optional


class MemeCoinPricePredictor:
    def __init__(self, model_dir='meme_coin_prediction_models'):
        """Initialize the predictor with saved models"""
        # Get the absolute path to the model directory
        import os
        
        # Try multiple possible locations
        possible_paths = [
            os.path.join(os.getcwd(), model_dir),
            os.path.join(os.path.dirname(__file__), model_dir),
            f'/workspaces/gcmemepredict/{model_dir}',
            model_dir
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'ensemble_model.pkl')):
                model_path = path
                print(f"✓ Found model directory: {path}")
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Model files not found in any of: {possible_paths}")
        
        # Load model
        self.model = joblib.load(os.path.join(model_path, 'ensemble_model.pkl'))
        
        # Load scalers
        self.scaler = joblib.load(os.path.join(model_path, 'feature_scaler.pkl'))
        
        # Load configuration
        with open(os.path.join(model_path, 'model_config.json')) as f:
            self.config = json.load(f)
        
        self.feature_names = self.config['feature_names']
            self.n_features = self.config['n_features']
            
        except FileNotFoundError as e:
            print(f"✗ Error: Model files not found in {model_dir}")
            print(f"  {e}")
            raise
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model configuration and performance metrics
        """
        return {
            'model_type': self.config['model_type'],
            'base_models': self.config['models'],
            'weights': self.config['weights'],
            'test_r2_score': self.config['test_r2'],
            'test_rmse': self.config['test_rmse'],
            'test_mae': self.config['test_mae'],
            'test_mape': f"{self.config['test_mape']:.4f}%",
            'n_features': self.n_features,
            'training_samples': self.config['training_samples'],
            'test_samples': self.config['test_samples'],
            'target_coin': self.config['target_coin'],
            'prediction_horizon': self.config['prediction_horizon']
        }
    
    def validate_features(self, features_dict: Dict) -> Tuple[bool, str]:
        """
        Validate that all required features are present.
        
        Args:
            features_dict (dict): Dictionary of feature values
            
        Returns:
            tuple: (is_valid, message)
        """
        missing_features = set(self.feature_names) - set(features_dict.keys())
        
        if missing_features:
            return False, f"Missing features: {missing_features}"
        
        # Check for NaN values
        for feat, value in features_dict.items():
            if pd.isna(value):
                return False, f"Feature '{feat}' has NaN value"
        
        return True, "All features valid"
    
    def predict_price(self, features_dict: Dict, 
                     return_std: bool = False) -> Tuple[float, Optional[float]]:
        """
        Predict next day's DOGE price given current features.
        
        Args:
            features_dict (dict): Dictionary of feature values
            return_std (bool): Whether to return prediction confidence (std dev)
            
        Returns:
            tuple: (predicted_price, std_dev) or (predicted_price,) if return_std=False
        """
        # Validate features
        is_valid, message = self.validate_features(features_dict)
        if not is_valid:
            raise ValueError(f"Feature validation failed: {message}")
        
        try:
            # Create feature array in correct order
            X = np.array([features_dict[feat] for feat in self.feature_names]).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction using ensemble
            prediction = self.model.predict(X_scaled)[0]
            
            # Get individual model predictions for confidence estimation
            if return_std:
                rf_pred = self.rf_model.predict(X_scaled)[0]
                gb_pred = self.gb_model.predict(X_scaled)[0]
                ab_pred = self.ab_model.predict(X_scaled)[0]
                
                # Calculate standard deviation across models
                predictions = np.array([rf_pred, gb_pred, ab_pred])
                std_dev = predictions.std()
                
                return prediction, std_dev
            
            return prediction, None
        
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            raise
    
    def predict_batch(self, features_list: List[Dict]) -> np.ndarray:
        """
        Make batch predictions for multiple samples.
        
        Args:
            features_list (list): List of feature dictionaries
            
        Returns:
            np.ndarray: Array of predicted prices
        """
        predictions = []
        
        for features_dict in features_list:
            try:
                pred, _ = self.predict_price(features_dict)
                predictions.append(pred)
            except Exception as e:
                print(f"✗ Error predicting sample: {e}")
                predictions.append(np.nan)
        
        return np.array(predictions)
    
    def fetch_live_data(self, coin_id: str = 'dogecoin', 
                       days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch live cryptocurrency data from CoinGecko API.
        
        Args:
            coin_id (str): CoinGecko coin ID (default: 'dogecoin')
            days (int): Number of days of historical data
            
        Returns:
            pd.DataFrame: DataFrame with price, volume, market_cap
        """
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract data
            prices = data['prices']
            volumes = data['total_volumes']
            market_caps = data['market_caps']
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(p[0]/1000) for p in prices],
                'price': [p[1] for p in prices],
                'volume': [v[1] for v in volumes],
                'market_cap': [m[1] for m in market_caps]
            })
            
            df['date'] = df['timestamp'].dt.date
            
            print(f"✓ Fetched {len(df)} records for {coin_id}")
            return df
        
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return None
    
    def create_features_from_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw market data.
        
        Args:
            df (pd.DataFrame): DataFrame with price, volume, market_cap
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        data.set_index('date', inplace=True)
        
        # Temporal Features
        data['day_of_week'] = data.index.dayofweek
        data['day_of_month'] = data.index.day
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Price Momentum Features
        data['price_change'] = data['price'].diff()
        data['price_pct_change'] = data['price'].pct_change() * 100
        data['price_volatility_7d'] = data['price'].rolling(window=7).std()
        data['price_volatility_14d'] = data['price'].rolling(window=14).std()
        
        # Moving Averages
        data['sma_7'] = data['price'].rolling(window=7).mean()
        data['sma_14'] = data['price'].rolling(window=14).mean()
        data['sma_30'] = data['price'].rolling(window=30).mean()
        data['ema_7'] = data['price'].ewm(span=7, adjust=False).mean()
        data['ema_14'] = data['price'].ewm(span=14, adjust=False).mean()
        
        # Volume Features
        data['volume_change'] = data['volume'].diff()
        data['volume_sma_7'] = data['volume'].rolling(window=7).mean()
        data['price_volume_ratio'] = data['price'] / (data['volume'] + 1)
        
        # Market Cap Features
        data['market_cap_change'] = data['market_cap'].diff()
        data['market_cap_volatility'] = data['market_cap'].rolling(window=7).std()
        
        # Ratio Features
        data['volume_to_market_cap'] = data['volume'] / (data['market_cap'] + 1)
        data['price_to_market_cap_ratio'] = (data['price'] * 1e9) / (data['market_cap'] + 1)
        
        # High/Low Range
        data['price_high_low_ratio'] = (data['price'].rolling(window=7).max() / 
                                       (data['price'].rolling(window=7).min() + 1e-8))
        
        # Lagged Features
        for lag in [1, 3, 7]:
            data[f'price_lag_{lag}'] = data['price'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        # Remove NaN values
        data = data.dropna()
        
        return data
    
    def predict_from_live_data(self, coin_id: str = 'dogecoin',
                              days: int = 30) -> Optional[float]:
        """
        End-to-end prediction pipeline: fetch data -> create features -> predict.
        
        Args:
            coin_id (str): CoinGecko coin ID
            days (int): Number of days of historical data to fetch
            
        Returns:
            float: Predicted next day price
        """
        print(f"\n{'='*70}")
        print(f"LIVE PREDICTION PIPELINE FOR {coin_id.upper()}")
        print(f"{'='*70}")
        
        # Step 1: Fetch live data
        print("\n[1/4] Fetching live data...")
        df = self.fetch_live_data(coin_id, days)
        if df is None:
            return None
        
        # Step 2: Create features
        print("[2/4] Creating engineered features...")
        df_features = self.create_features_from_data(df)
        print(f"✓ Created {len(df_features.columns)} features")
        
        # Step 3: Get latest features
        print("[3/4] Extracting latest features...")
        latest_features = df_features.iloc[-1]
        features_dict = {feat: latest_features[feat] 
                        for feat in self.feature_names 
                        if feat in latest_features.index}
        
        # Step 4: Make prediction
        print("[4/4] Making prediction...")
        try:
            prediction, std_dev = self.predict_price(features_dict, return_std=True)
            
            current_price = df['price'].iloc[-1]
            price_change = ((prediction - current_price) / current_price) * 100
            
            print(f"\n{'='*70}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*70}")
            print(f"Current Price:      ${current_price:.8f}")
            print(f"Predicted Price:    ${prediction:.8f}")
            print(f"Expected Change:    {price_change:+.2f}%")
            print(f"Confidence (Std):   ${std_dev:.8f}")
            print(f"{'='*70}\n")
            
            return prediction
        
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            return None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_1_basic_prediction():
    """Example 1: Basic prediction with sample features"""
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC PREDICTION WITH SAMPLE FEATURES")
    print("="*70)
    
    # Initialize predictor
    predictor = MemeCoinPricePredictor()
    
    # Print model info
    print("\nModel Information:")
    model_info = predictor.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Create sample features (you would get these from real data)
    sample_features = {
        'day_of_week': 3,
        'day_of_month': 15,
        'month': 6,
        'quarter': 2,
        'is_weekend': 0,
        'price_change': 0.001,
        'price_pct_change': 2.5,
        'price_volatility_7d': 0.005,
        'price_volatility_14d': 0.006,
        'sma_7': 0.082,
        'sma_14': 0.081,
        'sma_30': 0.080,
        'ema_7': 0.0825,
        'ema_14': 0.0815,
        'volume_change': 1000000,
        'volume_sma_7': 5000000,
        'price_volume_ratio': 0.00001,
        'market_cap_change': 100000000,
        'market_cap_volatility': 500000000,
        'volume_to_market_cap': 0.1,
        'price_to_market_cap_ratio': 0.0001,
        'price_high_low_ratio': 1.05,
        'price_lag_1': 0.081,
        'price_lag_3': 0.080,
        'price_lag_7': 0.079,
        'volume_lag_1': 5100000,
        'volume_lag_3': 4900000,
        'volume_lag_7': 4800000
    }
    
    # Make prediction
    print("\nMaking prediction...")
    try:
        prediction, std_dev = predictor.predict_price(sample_features, return_std=True)
        print(f"✓ Predicted Price: ${prediction:.8f}")
        print(f"  Confidence (Std): ${std_dev:.8f}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_2_live_prediction():
    """Example 2: Live prediction from real API data"""
    print("\n" + "="*70)
    print("EXAMPLE 2: LIVE PREDICTION FROM API DATA")
    print("="*70)
    
    predictor = MemeCoinPricePredictor()
    
    # Fetch live data and predict
    prediction = predictor.predict_from_live_data(coin_id='dogecoin', days=30)


def example_3_batch_prediction():
    """Example 3: Batch predictions for multiple samples"""
    print("\n" + "="*70)
    print("EXAMPLE 3: BATCH PREDICTIONS")
    print("="*70)
    
    predictor = MemeCoinPricePredictor()
    
    # Create sample features for multiple days
    sample_features_list = [
        {
            'day_of_week': i % 7,
            'day_of_month': 15 + i,
            'month': 6,
            'quarter': 2,
            'is_weekend': 1 if (i % 7 >= 5) else 0,
            'price_change': 0.001 * (i % 3),
            'price_pct_change': 2.5 + (i % 5),
            'price_volatility_7d': 0.005,
            'price_volatility_14d': 0.006,
            'sma_7': 0.082 + (i * 0.001),
            'sma_14': 0.081,
            'sma_30': 0.080,
            'ema_7': 0.0825,
            'ema_14': 0.0815,
            'volume_change': 1000000,
            'volume_sma_7': 5000000,
            'price_volume_ratio': 0.00001,
            'market_cap_change': 100000000,
            'market_cap_volatility': 500000000,
            'volume_to_market_cap': 0.1,
            'price_to_market_cap_ratio': 0.0001,
            'price_high_low_ratio': 1.05,
            'price_lag_1': 0.081,
            'price_lag_3': 0.080,
            'price_lag_7': 0.079,
            'volume_lag_1': 5100000,
            'volume_lag_3': 4900000,
            'volume_lag_7': 4800000
        }
        for i in range(5)
    ]
    
    print(f"\nMaking {len(sample_features_list)} predictions...")
    predictions = predictor.predict_batch(sample_features_list)
    
    print("\nBatch Predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Day {i+1}: ${pred:.8f}")


def example_4_model_comparison():
    """Example 4: Compare ensemble vs individual models"""
    print("\n" + "="*70)
    print("EXAMPLE 4: MODEL COMPARISON (ENSEMBLE VS INDIVIDUAL)")
    print("="*70)
    
    predictor = MemeCoinPricePredictor()
    
    # Fetch live data
    print("\nFetching live data...")
    df = predictor.fetch_live_data('dogecoin', days=30)
    if df is None:
        return
    
    # Create features
    print("Creating features...")
    df_features = predictor.create_features_from_data(df)
    latest_features = df_features.iloc[-1]
    
    # Create feature array
    X = np.array([latest_features[feat] for feat in predictor.feature_names]).reshape(1, -1)
    X_scaled = predictor.scaler.transform(X)
    
    # Get predictions from all models
    ensemble_pred = predictor.model.predict(X_scaled)[0]
    rf_pred = predictor.rf_model.predict(X_scaled)[0]
    gb_pred = predictor.gb_model.predict(X_scaled)[0]
    ab_pred = predictor.ab_model.predict(X_scaled)[0]
    
    current_price = df['price'].iloc[-1]
    
    print(f"\n{'='*70}")
    print(f"MODEL PREDICTIONS COMPARISON")
    print(f"{'='*70}")
    print(f"Current Price:              ${current_price:.8f}")
    print(f"{'='*70}")
    print(f"Random Forest:              ${rf_pred:.8f} ({((rf_pred-current_price)/current_price)*100:+.2f}%)")
    print(f"Gradient Boosting:          ${gb_pred:.8f} ({((gb_pred-current_price)/current_price)*100:+.2f}%)")
    print(f"AdaBoost:                   ${ab_pred:.8f} ({((ab_pred-current_price)/current_price)*100:+.2f}%)")
    print(f"{'='*70}")
    print(f"Ensemble (Weighted Avg):    ${ensemble_pred:.8f} ({((ensemble_pred-current_price)/current_price)*100:+.2f}%)")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MEME COIN PRICE PREDICTION DEPLOYMENT" + " "*17 + "║")
    print("║" + " "*15 + "Production Environment Examples" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\nChoose an example to run:")
    print("  1. Basic prediction with sample features")
    print("  2. Live prediction from API data")
    print("  3. Batch predictions")
    print("  4. Model comparison (Ensemble vs Individual)")
    print("  0. Exit")
    
    choice = input("\nEnter your choice (0-4): ").strip()
    
    try:
        if choice == '1':
            example_1_basic_prediction()
        elif choice == '2':
            example_2_live_prediction()
        elif choice == '3':
            example_3_batch_prediction()
        elif choice == '4':
            example_4_model_comparison()
        elif choice == '0':
            print("Exiting...")
        else:
            print("Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
