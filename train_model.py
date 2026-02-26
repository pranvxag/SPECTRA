"""
train_model_simple.py - Simple training script for Student Performance Predictor
ASCII-only version (guaranteed to work on any system)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

print("=" * 50)
print("STUDENT PERFORMANCE PREDICTOR - TRAINING")
print("=" * 50)

# Create models directory
os.makedirs('models', exist_ok=True)
print("[OK] Created models folder")

# Generate synthetic data
print("\n[STEP 1] Generating training data...")
np.random.seed(42)
n_samples = 1000

# Create features
data = {
    'studytime': np.random.randint(1, 5, n_samples),
    'failures': np.random.randint(0, 4, n_samples),
    'absences': np.random.randint(0, 30, n_samples),
    'goout': np.random.randint(1, 6, n_samples),
    'health': np.random.randint(1, 6, n_samples),
    'age': np.random.randint(15, 23, n_samples),
}

# Create target
df = pd.DataFrame(data)
df['score'] = (12 + df['studytime']*1.5 - df['failures']*3 
               - df['absences']/10 - df['goout']/2)
df['score'] = np.clip(df['score'], 0, 20)
df['target'] = (df['score'] >= 10).astype(int)

print(f"   - Created {n_samples} student records")
print(f"   - Features: {', '.join(data.keys())}")

# Prepare data
print("\n[STEP 2] Preparing data...")
X = df[['studytime', 'failures', 'absences', 'goout', 'health', 'age']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   - Training samples: {len(X_train)}")
print(f"   - Testing samples: {len(X_test)}")

# Train model
print("\n[STEP 3] Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
print("   - Model training complete")

# Evaluate
print("\n[STEP 4] Evaluating model...")
accuracy = model.score(X_test, y_test)
print(f"   - Model accuracy: {accuracy:.2%}")

# Save model
print("\n[STEP 5] Saving model...")
model_info = {
    'model': model,
    'name': 'Random Forest',
    'accuracy': accuracy,
    'features': X.columns.tolist()
}
joblib.dump(model_info, 'models/best_model.pkl')
print("   - Model saved to: models/best_model.pkl")

# Save preprocessor
preprocessor = {'feature_names': X.columns.tolist()}
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("   - Preprocessor saved to: models/preprocessor.pkl")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
print("\nNext: Run 'streamlit run app/app.py' to start the app")