"""
Configuration module for Student Performance Predictor
"""

import os
import json

class Config:
    """Configuration class for managing project settings"""
    
    def __init__(self, config_path='config.json'):
        # Default configuration
        self.config = {
            'random_state': 42,
            'test_size': 0.2,
            'target_column': 'G3',
            'drop_columns': ['G1', 'G2'],
            'risk_threshold': 10,
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6
                }
            }
        }
        
        # Try to load from file if it exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.json'):
                        loaded_config = json.load(f)
                        self.config.update(loaded_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
    
    def get(self, key, default=None):
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def get_model_params(self, model_name):
        """Get hyperparameters for specific model"""
        models = self.config.get('models', {})
        return models.get(model_name, {})  # ← NO triple quotes here!