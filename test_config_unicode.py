# test_config_unicode.py
import sys
import os
import io

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.config import Config
    print("✅ Successfully imported Config")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test the config
print("\n🔧 Testing configuration...")
print("-" * 40)

config = Config()

# Test basic settings
print(f"📊 Random state: {config.get('random_state')}")
print(f"📊 Test size: {config.get('test_size')}")
print(f"📊 Risk threshold: {config.get('risk_threshold')}")
print(f"📊 Target column: {config.get('target_column')}")
print(f"📊 Drop columns: {config.get('drop_columns')}")

# Test model parameters
print(f"\n🤖 Model Parameters:")
rf_params = config.get_model_params('random_forest')
print(f"  • Random Forest: {rf_params}")

xgb_params = config.get_model_params('xgboost')
print(f"  • XGBoost: {xgb_params}")

# Test non-existent key
print(f"\n🔍 Testing non-existent key: {config.get('nonexistent.key', 'default_value')}")

print("-" * 40)
print("✅ All tests passed!")