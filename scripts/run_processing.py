# Script chạy ETL processing
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.etl_pipeline import ETLPipeline
from src.utils.config import ConfigManager
import datetime
import glob

# Load config
config_manager = ConfigManager()
config = config_manager.load_config('config/config.yaml')

# Khởi tạo ETL pipeline
etl = ETLPipeline(config)

# Tìm file raw mới nhất
raw_files = glob.glob('data/raw/churn_data_*.parquet')
if not raw_files:
    raise FileNotFoundError("No raw data files found. Please run ingestion first.")
input_path = max(raw_files)  # Lấy file mới nhất
print(f"Using input file: {input_path}")

# Tạo output path
output_path = f"data/processed/churn_processed_{datetime.datetime.now().strftime('%Y%m%d')}.parquet"

# Chạy ETL pipeline
metadata = etl.run_pipeline(
    input_path=input_path,
    output_path=output_path,
    schema_file="config/schemas/churn_schema.json",
    run_quality_checks=True,
    run_feature_engineering=True
)

print("Processing completed successfully!")
print(f"Output file: {metadata['output_file']}")
print(f"Rows processed: {metadata.get('row_count', 'N/A')}")



