# Script train model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.model_trainer import ModelTrainer
from src.utils.config import ConfigManager
import pandas as pd
import glob

# Load config
config_manager = ConfigManager()
config = config_manager.load_config('config/config.yaml')

# Tìm file processed mới nhất
processed_files = glob.glob('data/processed/churn_processed_*.parquet')
if not processed_files:
    raise FileNotFoundError("No processed data files found. Please run processing first.")
input_file = max(processed_files)  # Lấy file mới nhất
print(f"Training with data from: {input_file}")

# Đọc processed data
df = pd.read_parquet(input_file)
print(f"Loaded {len(df)} rows for training")

# Khởi tạo trainer
trainer = ModelTrainer(config)

# Chuẩn bị dữ liệu
print("Preparing data...")
X, y, feature_columns = trainer.prepare_data(df)
print(f"[OK] Prepared {len(feature_columns)} features for {len(X)} samples")
print(f"  Churn rate: {y.mean():.2%}")

# Train model
print("\nTraining model (this may take a few minutes)...")
results = trainer.train_model(X, y, algorithm="xgboost")

# Hiển thị kết quả
print("\n" + "="*50)
print("TRAINING RESULTS")
print("="*50)
print(f"Algorithm: {results.get('algorithm', 'N/A')}")
print(f"Model saved to: {results.get('model_path', 'N/A')}")

test_metrics = results.get('test_metrics', {})
print("\nTest Metrics:")
print(f"  Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
print(f"  Precision: {test_metrics.get('precision', 0):.4f}")
print(f"  Recall:    {test_metrics.get('recall', 0):.4f}")
print(f"  F1-Score:  {test_metrics.get('f1', 0):.4f}")
print(f"  ROC-AUC:   {test_metrics.get('roc_auc', 0):.4f}")

print("\n[SUCCESS] Model training completed successfully!")



