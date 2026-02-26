import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
from src.config import Config

class ModelTrainer:
    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
    def get_models(self, task_type='classification'):
        """Define models based on task type"""
        if task_type == 'classification':
            return {
                'logistic_regression': LogisticRegression(
                    C=self.config.get('models.logistic_regression.C', 1.0),
                    max_iter=self.config.get('models.logistic_regression.max_iter', 1000),
                    random_state=self.config.get('random_state')
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=self.config.get('models.random_forest.n_estimators', 100),
                    max_depth=self.config.get('models.random_forest.max_depth', 10),
                    min_samples_split=self.config.get('models.random_forest.min_samples_split', 5),
                    random_state=self.config.get('random_state')
                ),
                'xgboost': XGBClassifier(
                    n_estimators=self.config.get('models.xgboost.n_estimators', 100),
                    max_depth=self.config.get('models.xgboost.max_depth', 6),
                    learning_rate=self.config.get('models.xgboost.learning_rate', 0.1),
                    random_state=self.config.get('random_state'),
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            }
        else:  # regression
            return {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.config.get('random_state')
                ),
                'xgboost': XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.config.get('random_state')
                )
            }
    
    def prepare_binary_target(self, y, threshold=None):
        """Convert continuous grades to binary (pass/fail)"""
        if threshold is None:
            threshold = self.config.get('risk_threshold', 10)
        return (y >= threshold).astype(int)
    
    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train multiple classification models"""
        print("Training classification models...")
        
        # Prepare binary target
        y_train_binary = self.prepare_binary_target(y_train)
        y_test_binary = self.prepare_binary_target(y_test)
        
        models = self.get_models('classification')
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train_binary)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_binary, y_pred)
            precision = precision_score(y_test_binary, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_binary, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_binary, y_pred, average='weighted', zero_division=0)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            # Track best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_model_name = name
        
        return self.results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """Train multiple regression models"""
        print("Training regression models...")
        
        models = self.get_models('regression')
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R² Score: {r2:.4f}")
            
            # Track best model (lower RMSE is better)
            if name == 'linear_regression' or rmse < self.best_score:
                self.best_score = rmse if name != 'linear_regression' else rmse
                self.best_model = model
                self.best_model_name = name
        
        return self.results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """Perform hyperparameter tuning using GridSearchCV"""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        y_train_binary = self.prepare_binary_target(y_train)
        
        if model_name == 'random_forest':
            model = RandomForestClassifier(random_state=self.config.get('random_state'))
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'xgboost':
            model = XGBClassifier(random_state=self.config.get('random_state'), 
                                 use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
        else:
            print(f"Tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train_binary)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model(self, path='models/best_model.pkl'):
        """Save the best model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_info = {
            'model': self.best_model,
            'name': self.best_model_name,
            'score': self.best_score,
            'all_results': self.results
        }
        
        joblib.dump(model_info, path)
        print(f"Best model ({self.best_model_name}) saved to {path} with score: {self.best_score:.4f}")
        
        return path
    
    def load_model(self, path='models/best_model.pkl'):
        """Load a saved model"""
        model_info = joblib.load(path)
        self.best_model = model_info['model']
        self.best_model_name = model_info['name']
        self.best_score = model_info['score']
        self.results = model_info['all_results']
        
        print(f"Model loaded from {path}")
        return self.best_model