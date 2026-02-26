import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from src.config import Config
import joblib

class DataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load raw data from CSV"""
        return pd.read_csv(filepath, sep=';')  # UCI dataset uses semicolon delimiter
    
    def preprocess(self, df):
        """Main preprocessing pipeline"""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Drop intermediate grades if configured
        drop_cols = self.config.get('drop_columns', [])
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Separate features and target
        target_col = self.config.get('target_column')
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values (if any)
        X = X.fillna(X.mean())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def split_data(self, X, y):
        """Split into train and test sets"""
        test_size = self.config.get('test_size')
        random_state = self.config.get('random_state')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Save preprocessor objects"""
        preprocessor = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(preprocessor, path)
        print(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Load preprocessor objects"""
        preprocessor = joblib.load(path)
        self.label_encoders = preprocessor['label_encoders']
        self.scaler = preprocessor['scaler']
        self.feature_names = preprocessor['feature_names']
        return preprocessor
    
    def run_pipeline(self, raw_data_path, save=True):
        """Run complete preprocessing pipeline"""
        print("Loading data...")
        df = self.load_data(raw_data_path)
        
        print("Preprocessing data...")
        X, y = self.preprocess(df)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        if save:
            self.save_preprocessor()
            
            # Save processed data
            processed_dir = 'data/processed'
            os.makedirs(processed_dir, exist_ok=True)
            
            pd.DataFrame(X_train, columns=self.feature_names).to_csv(
                f'{processed_dir}/X_train.csv', index=False
            )
            pd.DataFrame(X_test, columns=self.feature_names).to_csv(
                f'{processed_dir}/X_test.csv', index=False
            )
            pd.Series(y_train, name='target').to_csv(
                f'{processed_dir}/y_train.csv', index=False
            )
            pd.Series(y_test, name='target').to_csv(
                f'{processed_dir}/y_test.csv', index=False
            )
            
            print("Processed data saved to data/processed/")
        
        return X_train, X_test, y_train, y_test