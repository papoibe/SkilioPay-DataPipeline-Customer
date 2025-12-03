

# Standard library imports
import pandas as pd 
import pyarrow as pa  
import pyarrow.parquet as pq 
from pathlib import Path  
from typing import Dict, Any, Optional  
import logging  
from datetime import datetime  
import json  

# Internal imports - các utility modules của project
from ..utils.config import ConfigManager  # Load config từ YAML
from ..utils.logging_config import get_logger  # Setup structured logging
from ..utils.data_validation import DataValidator  # Validate data theo schema

# Khởi tạo logger cho module này
logger = get_logger(__name__)


class CSVIngestion:
    # Class này xử lý việc đọc CSV file, validate dữ liệu, và convert sang Parquet format.
    def __init__(self, config: Dict[str, Any]):
        # Lưu config để dùng trong các method khác
        self.config = config
        # Khởi tạo DataValidator để validate data theo schema
        self.validator = DataValidator()
        # Timestamp khi khởi tạo object (dùng cho metadata)
        self.ingestion_timestamp = datetime.now().isoformat()
        
    #  xử lý việc đọc CSV file, validate dữ liệu, và convert sang Parquet format.
    def ingest_csv(
        self, 
        input_path: str,  # Path to input CSV file
        output_path: str,
        schema_file: Optional[str] = None,
        validate_data: bool = True
    ) -> Dict[str, Any]:

        try:
            logger.info(f"Starting CSV ingestion from {input_path}")
            
            # Bước 1: Đọc CSV file vào DataFrame
            # Sử dụng pandas để đọc CSV với config từ config.yaml
            df = self._read_csv(input_path)
            logger.info(f"Successfully read {len(df)} rows from CSV")
            
            # Bước 2: Load schema nếu được cung cấp
            # Schema file (JSON) định nghĩa structure và validation rules cho data
            schema = None
            if schema_file and validate_data:
                schema = self._load_schema(schema_file)
                logger.info("Schema loaded for validation")
            
            # Bước 3: Validate data theo schema
            # Kiểm tra: data types, ranges, required fields, enum values, patterns
            if schema and validate_data:
                validation_result = self.validator.validate_dataframe(df, schema)
                if not validation_result["is_valid"]:
                    # Nếu validation fail, log errors và raise exception
                    logger.error(f"Data validation failed: {validation_result['errors']}")
                    raise ValueError("Data validation failed")
                logger.info("Data validation passed")
            
            # Bước 4: Thêm metadata columns
            # Metadata giúp track: khi nào ingest, từ file nào, row number
            df = self._add_metadata(df)
            
            # Bước 5: Convert DataFrame sang Parquet format
            # Parquet: columnar storage, compression tốt, đọc nhanh hơn CSV
            parquet_path = self._convert_to_parquet(df, output_path)
            logger.info(f"Successfully converted to Parquet: {parquet_path}")
            
            # Bước 6: Generate metadata về quá trình ingestion
            # Metadata này được return để tracking và monitoring
            metadata = self._generate_metadata(df, input_path, parquet_path)
            
            logger.info("CSV ingestion completed successfully")
            return metadata
            
        except Exception as e:
            # Log error và re-raise để caller có thể handle
            logger.error(f"CSV ingestion failed: {str(e)}")
            raise
    
    #  đọc CSV file vào DataFrame
    def _read_csv(self, input_path: str) -> pd.DataFrame:
        # Lấy config cho CSV từ config.yaml
        csv_config = self.config.get("data_sources", {}).get("csv", {})
        
        # Đọc CSV với config từ file config
        return pd.read_csv(
            input_path,
            delimiter=csv_config.get("delimiter", ","),  # Default: comma
            encoding=csv_config.get("encoding", "utf-8"),  # Default: UTF-8
            low_memory=False  # Đọc toàn bộ để infer dtypes chính xác
        )
    
    #  load schema từ JSON fil
    def _load_schema(self, schema_file: str) -> Dict[str, Any]:
        # Đọc JSON schema file
        with open(schema_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    #  thêm metadata columns vào DataFrame
    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        # Copy để không modify original dataframe
        df = df.copy()
        
        # Timestamp khi data được ingest (ISO format)
        df['_ingestion_timestamp'] = self.ingestion_timestamp
        
        # Tên file nguồn (chỉ lấy filename, không có path)
        df['_source_file'] = Path(self.config.get("data_sources", {}).get("csv", {}).get("input_path", "")).name
        
        # Row number trong file gốc (0-indexed)
        df['_row_number'] = range(len(df))
        
        return df
    

    #  convert DataFrame sang Parquet format
    def _convert_to_parquet(self, df: pd.DataFrame, output_path: str) -> str:
        # Tạo thư mục output nếu chưa tồn tại
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert pandas DataFrame sang PyArrow Table
        # PyArrow Table hiệu quả hơn cho việc ghi Parquet
        table = pa.Table.from_pandas(df)
        
        # Đảm bảo file path có extension .parquet
        parquet_path = f"{output_path.rstrip('.parquet')}.parquet"
        
        # Ghi Parquet file với compression
        pq.write_table(
            table, 
            parquet_path,
            compression='snappy',  # Snappy: balance giữa tốc độ và tỷ lệ nén
            use_deprecated_int96_timestamps=False  # Dùng timestamp format mới
        )
        
        return parquet_path
    
    def _generate_metadata(self, df: pd.DataFrame, input_path: str, output_path: str) -> Dict[str, Any]:
       
        # - Tracking: biết được quá trình ingestion thành công hay không
        # - Monitoring: số rows, columns, file size
        # - Debugging: biết được structure của data
        # - Logging: ghi lại vào logs hoặc database
        return {
            "ingestion_timestamp": self.ingestion_timestamp,  # Khi nào ingest
            "input_file": input_path,  # File CSV input
            "output_file": output_path,  # File Parquet output
            "row_count": len(df),  # Số dòng dữ liệu
            "column_count": len(df.columns),  # Số cột
            "file_size_bytes": Path(output_path).stat().st_size if Path(output_path).exists() else 0,  # Dung lượng file
            "columns": list(df.columns),  # Danh sách tên columns
            "dtypes": df.dtypes.to_dict(),  # Data types của từng column
            "status": "success"  # Trạng thái: success hoặc failed
        }


def main():

    import argparse
    
    # Setup argument parser cho CLI
    parser = argparse.ArgumentParser(description="CSV Data Ingestion")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    parser.add_argument("--schema", help="JSON schema file for validation")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--no-validate", action="store_true", help="Skip data validation")
    
    # Parse arguments từ command line
    args = parser.parse_args()
    
    # Load configuration từ YAML file
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Khởi tạo CSVIngestion object với config
    ingestion = CSVIngestion(config)
    
    # Chạy ingestion process
    metadata = ingestion.ingest_csv(
        input_path=args.input,
        output_path=args.output,
        schema_file=args.schema,
        validate_data=not args.no_validate  # Validate nếu không có flag --no-validate
    )
    
    # In kết quả ra console
    print(f"Ingestion completed: {metadata}")


if __name__ == "__main__":
    main()
