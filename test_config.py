# test_config.py
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.config import Config
    print("[OK] Successfully imported Config")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Test the config
print("\n" + "="*50)
print("TESTING CONFIGURATION")
print("="*50)

config = Config()

# Test basic settings
print(f"\n📊 Basic Settings:")
print(f"  • Random state: {config.get('random_state')}")
print(f"  • Test size: {config.get('test_size')}")
print(f"  • Risk threshold: {config.get('risk_threshold')}")
print(f"  • Target column: {config.get('target_column')}")
print(f"  • Drop columns: {config.get('drop_columns')}")

# Test model parameters
print(f"\n🤖 Model Parameters:")
rf_params = config.get_model_params('random_forest')
print(f"  • Random Forest: {rf_params}")

xgb_params = config.get_model_params('xgboost')
print(f"  • XGBoost: {xgb_params}")

# Test non-existent key
print(f"\n🔍 Testing default values:")
result = config.get('nonexistent.key', 'default_value')
print(f"  • Non-existent key returns: '{result}'")

print("\n" + "="*50)
print("✓ CONFIG TEST PASSED!")
print("="*50)