# Main ETL pipeline for data processing
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import json

from utils.config import ConfigManager
from utils.logging_config import get_logger
from utils.data_validation import DataValidator
from utils.cache_manager import PipelineCache
from .feature_engineering import FeatureEngineer
from .data_quality import DataQualityChecker

logger = get_logger(__name__)


class ETLPipeline:
    
    def __init__(self, config: Dict[str, Any]):

        self.config = config
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer(config)
        self.quality_checker = DataQualityChecker()
        self.processing_timestamp = datetime.now().isoformat()
        # Cache cho các bước tốn kém (feature engineering)
        self.cache = PipelineCache(cache_dir="cache/etl", default_ttl=24 * 3600)
        
    #  run ETL pipeline
    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        schema_file: Optional[str] = None, # schema file for validation
        run_quality_checks: bool = True, # whether to run data quality checks
        run_feature_engineering: bool = True, # whether to run feature engineering
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Starting ETL pipeline from {input_path}")
            
            # bước 1: Extract - Load data
            df = self._extract_data(input_path)
            logger.info(f"Successfully loaded {len(df)} rows")
            
            # bước 2: Transform - Process data
            df_processed = self._transform_data(
                df, 
                schema_file,  # schema file for validation
                run_quality_checks,  # whether to run data quality checks
                run_feature_engineering  # whether to run feature engineering
            )
            logger.info(f"Data transformation completed: {len(df_processed)} rows")
            
            # bước 3: Load - Save processed data
            output_file = self._load_data(df_processed, output_path)
            logger.info(f"Processed data saved to {output_file}")
            
            # Generate processing metadata
            metadata = self._generate_metadata(df, df_processed, input_path, output_file)
            
            logger.info("ETL pipeline completed successfully")
            return metadata
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise
    
    #  extract data from input file
    def _extract_data(self, input_path: str) -> pd.DataFrame:
        if input_path.endswith('.parquet'):
            return pd.read_parquet(input_path)
        elif input_path.endswith('.csv'):
            return pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
    #  xử lý dữ liệu thô thành dữ liệu đã xử lý
    def _transform_data(
        self,
        df: pd.DataFrame,
        schema_file: Optional[str],
        run_quality_checks: bool,
        run_feature_engineering: bool
    ) -> pd.DataFrame:
        df_processed = df.copy() 
        
        # bước 1: Data validation
        if schema_file:
            schema = self._load_schema(schema_file) #
            validation_result = self.validator.validate_dataframe(df_processed, schema) # validate data against schema
            if not validation_result["is_valid"]:
                logger.warning(f"Data validation issues: {validation_result['errors']}")
                # Continue processing but log warnings
        
        # bước 2: Data quality checks
        if run_quality_checks:
            quality_report = self.quality_checker.check_data_quality(df_processed) # check data quality
            logger.info(f"Data quality report: {quality_report}")
        
        # bước 3: Data cleaning
        df_processed = self._clean_data(df_processed) # clean data
        logger.info("Data cleaning completed")
        
        # bước 4: Feature engineering (có cache)
        if run_feature_engineering:
            cached = self.cache.get(operation="feature_engineering", data=df_processed)
            if cached is not None:
                df_processed = cached
                logger.info("Feature engineering loaded from cache")
            else:
                df_processed = self.feature_engineer.engineer_features(df_processed)
                self.cache.set(operation="feature_engineering", value=df_processed, data=df_processed)
                logger.info("Feature engineering completed and cached")
        
        # bướ 5: Data normalization
        df_processed = self._normalize_data(df_processed)
        logger.info("Data normalization completed")
        
        # bước 6: Add processing metadata
        df_processed = self._add_processing_metadata(df_processed)
        
        return df_processed
    

    #  load schema from JSON file
    def _load_schema(self, schema_file: str) -> Dict[str, Any]:
        with open(schema_file, 'r', encoding='utf-8') as f: # open schema file
            return json.load(f) # load schema from JSON file
    
    #  clean and preprocess data
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_cleaned)} duplicate rows")
        
        # Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned)
        
        # Handle outliers
        df_cleaned = self._handle_outliers(df_cleaned)
        
        # Data type corrections
        df_cleaned = self._correct_data_types(df_cleaned)
        
        return df_cleaned
    
    #  handle missing values in the dataset
    #  follow 2 strategies: impute: điền giá trị thay thế; Drop: loại bỏ các hàng chứa giá trị thiếu 
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        # Get missing value strategy from config
        strategy = self.config.get("processing", {}).get("transformations", {}).get("missing_value_handling", "impute")
        
        if strategy == "impute": 
            # Numerical columns - điền giá trị thiếu bằng median (ổn hơn mean, không bị ảnh hưởng bới outlier như mean)
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns # get numerical columns
            for col in numerical_cols:
                if df_cleaned[col].isnull().any(): 
                    median_val = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
                    logger.info(f"Imputed missing values in {col} with median: {median_val}")
            
            # Categorical columns - dùng mode để điền giá trị thiếu 
            categorical_cols = df_cleaned.select_dtypes(include=['object']).columns # lấy danh sách object(string, cate)
            for col in categorical_cols:
                if df_cleaned[col].isnull().any():
                    mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "Unknown" # Nếu có giá trị thiếu → tính mode (giá trị xuất hiện nhiều nhất)Nếu cột rỗng hoàn toàn → dùng "Unknown"
                    df_cleaned[col].fillna(mode_val, inplace=True)
                    logger.info(f"Imputed missing values in {col} with mode: {mode_val}")
        
        elif strategy == "drop": # DÙNG KHI DỮ LIỆU ĐÃ SẠCH
            # Drop rows with any missing values
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            logger.info(f"Dropped {initial_rows - len(df_cleaned)} rows with missing values")
        
        return df_cleaned
    
    # tính outliers 
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        # Get numerical columns
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in ['churn_label', 'rfm_recency', 'rfm_frequency']:  # Skip target and categorical numerical
                continue
                
            # Use IQR method for outlier detection
            Q1 = df_cleaned[col].quantile(0.25) 
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            outliers_count = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
            if outliers_count > 0:
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped {outliers_count} outliers in {col}")
        
        return df_cleaned
    
    # chuyển đổi kiểu dữ liệu của các cột
    def _correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        # Convert user_id to string if it's not already
        if 'user_id' in df_cleaned.columns:
            df_cleaned['user_id'] = df_cleaned['user_id'].astype(str)
        
        # Convert churn_label to int
        if 'churn_label' in df_cleaned.columns:
            df_cleaned['churn_label'] = df_cleaned['churn_label'].astype(int)
        
        # Convert app_version_major to string
        if 'app_version_major' in df_cleaned.columns:
            df_cleaned['app_version_major'] = df_cleaned['app_version_major'].astype(str)
        
        return df_cleaned

    # chuẩn hóa (normalize) các cột số bằng StandardScaler
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_normalized = df.copy()
        # Get numerical columns for normalization
        numerical_cols = df_normalized.select_dtypes(include=[np.number]).columns
        target_cols = ['churn_label', 'rfm_recency', 'rfm_frequency', 'rfm_monetary'] # Loại bỏ các cột không được chuẩn hóa
        normalize_cols = [col for col in numerical_cols if col not in target_cols]
        
        if normalize_cols:
            # Initialize scaler
            scaler = StandardScaler()
            
            # Fit and transform
            df_normalized[normalize_cols] = scaler.fit_transform(df_normalized[normalize_cols])
            logger.info(f"Normalized {len(normalize_cols)} numerical columns")
        
        return df_normalized
    
    # thêm metadata vào dataframe
    def _add_processing_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        df_processed['_processing_timestamp'] = self.processing_timestamp
        df_processed['_processing_version'] = "1.0.0"
        return df_processed
    
    def _load_data(self, df: pd.DataFrame, output_path: str) -> str:
        # Create output directory if not exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to PyArrow Table for better performance
        table = pa.Table.from_pandas(df)
        
        # Write to Parquet with compression
        # removesuffix: chỉ xóa suffix chính xác, không xóa từng ký tự như rstrip
        parquet_path = f"{output_path.removesuffix('.parquet')}.parquet"
        pq.write_table(
            table, 
            parquet_path,
            compression='snappy',
            use_deprecated_int96_timestamps=False
        )
        
        return parquet_path
    
    # tạo metadata về quá trình xử lý dữ liệu
    def _generate_metadata(
        self, 
        df_original: pd.DataFrame, 
        df_processed: pd.DataFrame, 
        input_path: str, 
        output_path: str
    ) -> Dict[str, Any]:
        return {
            "processing_timestamp": self.processing_timestamp,
            "input_file": input_path,
            "output_file": output_path,
            "original_row_count": len(df_original),
            "processed_row_count": len(df_processed),
            "original_column_count": len(df_original.columns),
            "processed_column_count": len(df_processed.columns),
            "file_size_bytes": Path(output_path).stat().st_size if Path(output_path).exists() else 0,
            "columns_added": list(set(df_processed.columns) - set(df_original.columns)),
            "columns_removed": list(set(df_original.columns) - set(df_processed.columns)),
            "status": "success"
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ETL Pipeline")
    parser.add_argument("--input", required=True, help="Input Parquet file path")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    parser.add_argument("--schema", help="JSON schema file for validation")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--no-quality-checks", action="store_true", help="Skip data quality checks")
    parser.add_argument("--no-feature-engineering", action="store_true", help="Skip feature engineering")
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Initialize ETL pipeline
    etl = ETLPipeline(config)
    
    # Run pipeline
    metadata = etl.run_pipeline(
        input_path=args.input,
        output_path=args.output,
        schema_file=args.schema,
        run_quality_checks=not args.no_quality_checks,
        run_feature_engineering=not args.no_feature_engineering
    )
    
    print(f"ETL pipeline completed: {metadata}")


if __name__ == "__main__":
    main()
