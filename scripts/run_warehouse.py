# Script load data vào warehouse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.data_warehouse import ChurnDataWarehouse
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
print(f"Loading from: {input_file}")

# Khởi tạo data warehouse (không tạo bảng tự động, giả định đã có sẵn)
dw = ChurnDataWarehouse(config, create_tables=False)

# Đọc processed data
df = pd.read_parquet(input_file)
print(f"Loaded {len(df)} rows from processed file")

# Load vào bảng users_processed
print("Loading to users_processed table...")
dw.load_data(df, "users_processed", if_exists="replace")
print("[OK] Loaded to users_processed")

# Load features vào bảng features (nếu có)
# Lọc bỏ các cột metadata (bắt đầu bằng _) và chỉ giữ features
exclude_cols = [col for col in df.columns if col.startswith('_')]
feature_cols = [col for col in df.columns if col not in exclude_cols and col != 'churn_label']
features_df = df[['user_id'] + feature_cols].copy()

print("Loading to features table...")
dw.load_data(features_df, "features", if_exists="replace")
print("[OK] Loaded to features")

print(f"\n[SUCCESS] Successfully loaded {len(df)} rows to warehouse!")



