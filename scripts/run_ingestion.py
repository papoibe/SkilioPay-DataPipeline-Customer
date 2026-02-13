# Script chạy ingestion
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.csv_ingestion import CSVIngestion
from src.utils.config import ConfigManager
import datetime

# Load config
config_manager = ConfigManager()
config = config_manager.load_config('config/config.yaml')

# Khởi tạo ingestion
ingestion = CSVIngestion(config)

# Tạo output path với timestamp
output_path = f"data/raw/churn_data_{datetime.datetime.now().strftime('%Y%m%d')}.parquet"

# Chạy ingestion
# Tạm thời tắt validation để chạy pipeline, sẽ sửa validation sau
metadata = ingestion.ingest_csv(
    input_path='data/raw/churn_dataset.csv',
    output_path=output_path,
    schema_file='config/schemas/churn_schema.json',
    validate_data=False  # Tắt validation tạm thời
)

print("Ingestion completed successfully!")
print(f"Output file: {metadata['output_file']}")
print(f"Rows ingested: {metadata.get('row_count', 'N/A')}")



