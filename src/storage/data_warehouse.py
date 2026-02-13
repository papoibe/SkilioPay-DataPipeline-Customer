# Data Warehouse Module
# Xử lý lưu trữ và truy xuất dữ liệu cho PostgreSQL data warehouse
# Hỗ trợ tạo bảng, load dữ liệu, và query với indexing phù hợp

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
import json

from utils.logging_config import get_logger

logger = get_logger(__name__)


class DataWarehouse:
    # PostgreSQL data warehouse handler - xử lý kết nối và thao tác với data warehouse
    
    def __init__(self, config: Dict[str, Any]):
        # Khởi tạo kết nối data warehouse
        # Args:
        #   config: Dictionary chứa cấu hình (từ config.yaml)
        self.config = config
        self.connection_string = self._build_connection_string()
        self.engine = create_engine(self.connection_string)
        self.schema = config.get("storage", {}).get("database", {}).get("postgresql", {}).get("schema", "public")
        
    def _build_connection_string(self) -> str:
        # Xây dựng connection string cho PostgreSQL từ config
        # URL encode password để xử lý ký tự đặc biệt như @, #, %, etc.
        from urllib.parse import quote_plus
        db_config = self.config.get("storage", {}).get("database", {}).get("postgresql", {})
        
        username = quote_plus(db_config.get('username', 'postgres'))
        password = quote_plus(db_config.get('password', ''))
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 5432)
        database = db_config.get('database', 'postgres')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def create_tables(self, table_schemas: Dict[str, str]):
        # Tạo các bảng trong data warehouse
        # Args:
        #   table_schemas: Dictionary với key là tên bảng, value là SQL CREATE TABLE
        try:
            with self.engine.connect() as conn:
                for table_name, create_sql in table_schemas.items():
                    # Add schema prefix
                    full_table_name = f"{self.schema}.{table_name}"
                    create_sql = create_sql.replace(f"CREATE TABLE {table_name}", f"CREATE TABLE {full_table_name}")
                    
                    conn.execute(text(create_sql))
                    conn.commit()
                    logger.info(f"Created table: {full_table_name}")
                    
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    def load_data(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        if_exists: str = "replace",
        index: bool = False,
        chunksize: int = 10000
    ) -> bool:
        # Load DataFrame vào bảng PostgreSQL
        # Args:
        #   df: DataFrame cần load
        #   table_name: Tên bảng đích
        #   if_exists: Hành động nếu bảng đã tồn tại ('replace', 'append', 'fail')
        #   index: Có bao gồm index của DataFrame không
        #   chunksize: Số dòng ghi mỗi lần (batch size)
        # Returns:
        #   True nếu thành công
        try:
            full_table_name = f"{self.schema}.{table_name}"
            
            df.to_sql(
                table_name,
                self.engine,
                schema=self.schema,
                if_exists=if_exists,
                index=index,
                chunksize=chunksize,
                method='multi'
            )
            
            logger.info(f"Successfully loaded {len(df)} rows to {full_table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data to {table_name}: {str(e)}")
            raise
    
    def query_data(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        # Thực thi query SQL và trả về DataFrame
        # Args:
        #   query: Chuỗi SQL query
        #   params: Tham số cho query (optional)
        # Returns:
        #   DataFrame chứa kết quả query
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                logger.info(f"Query executed successfully, returned {len(df)} rows")
                return df
                
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    def create_indexes(self, table_name: str, indexes: List[Dict[str, Any]]):
        # Tạo indexes trên bảng
        # Args:
        #   table_name: Tên bảng
        #   indexes: List các định nghĩa index (dict với 'name', 'columns', 'type')
        try:
            with self.engine.connect() as conn:
                for index in indexes:
                    index_name = index['name']
                    columns = index['columns']
                    index_type = index.get('type', 'btree')
                    
                    if isinstance(columns, list):
                        columns_str = ', '.join(columns)
                    else:
                        columns_str = columns
                    
                    create_index_sql = f"""
                    CREATE INDEX IF NOT EXISTS {index_name} 
                    ON {self.schema}.{table_name} 
                    USING {index_type} ({columns_str})
                    """
                    
                    conn.execute(text(create_index_sql))
                    conn.commit()
                    logger.info(f"Created index: {index_name}")
                    
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        # Lấy thông tin về bảng (metadata)
        # Args:
        #   table_name: Tên bảng
        # Returns:
        #   Dictionary chứa metadata: columns, row_count, column_count
        try:
            query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_schema = '{self.schema}' 
            AND table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            
            df = self.query_data(query)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {self.schema}.{table_name}"
            count_df = self.query_data(count_query)
            row_count = count_df['row_count'].iloc[0] if not count_df.empty else 0
            
            return {
                "table_name": f"{self.schema}.{table_name}",
                "columns": df.to_dict('records'),
                "row_count": row_count,
                "column_count": len(df)
            }
            
        except Exception as e:
            logger.error(f"Failed to get table info: {str(e)}")
            raise
    
    def backup_table(self, table_name: str, backup_suffix: str = None) -> str:
        # Tạo backup của bảng
        # Args:
        #   table_name: Tên bảng cần backup
        #   backup_suffix: Hậu tố cho tên bảng backup (nếu None thì dùng timestamp)
        # Returns:
        #   Tên bảng backup đã tạo
        try:
            if backup_suffix is None:
                backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S") 
            
            backup_table_name = f"{table_name}_backup_{backup_suffix}"
            
            query = f"""
            CREATE TABLE {self.schema}.{backup_table_name} AS 
            SELECT * FROM {self.schema}.{table_name}
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(query))
                conn.commit()
            
            logger.info(f"Table backup created: {backup_table_name}")
            return backup_table_name
            
        except Exception as e:
            logger.error(f"Failed to backup table: {str(e)}")
            raise
    
    def optimize_table(self, table_name: str):
        # Tối ưu bảng (chạy VACUUM và ANALYZE)
        # Args:
        #   table_name: Tên bảng cần tối ưu
        try:
            with self.engine.connect() as conn:
                # VACUUM
                conn.execute(text(f"VACUUM {self.schema}.{table_name}"))
                
                # ANALYZE
                conn.execute(text(f"ANALYZE {self.schema}.{table_name}"))
                
                conn.commit()
            
            logger.info(f"Table optimized: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to optimize table: {str(e)}")
            raise


class ChurnDataWarehouse(DataWarehouse):
    # Data warehouse chuyên biệt cho dữ liệu churn prediction
    
    def __init__(self, config: Dict[str, Any], create_tables: bool = False):
        # Khởi tạo churn data warehouse
        # Args:
        #   config: Dictionary chứa cấu hình
        #   create_tables: Nếu True, tự động tạo bảng. Nếu False (mặc định), giả định bảng đã tồn tại
        super().__init__(config)
        if create_tables:
            self._create_churn_tables()
    
    def _create_churn_tables(self):
        # Tạo các bảng chuyên biệt cho churn prediction
        table_schemas = {
            "users_raw": """
                CREATE TABLE users_raw (
                    user_id VARCHAR(10) PRIMARY KEY,
                    age INTEGER,
                    country VARCHAR(50),
                    city VARCHAR(100),
                    reg_days INTEGER,
                    marketing_source VARCHAR(50),
                    sessions_30d INTEGER,
                    sessions_90d INTEGER,
                    avg_session_duration_90d DECIMAL(10,2),
                    median_pages_viewed_30d DECIMAL(10,2),
                    search_queries_30d INTEGER,
                    device_mix_ratio DECIMAL(5,3),
                    app_version_major VARCHAR(10),
                    orders_30d INTEGER,
                    orders_90d INTEGER,
                    orders_2024 INTEGER,
                    aov_2024 DECIMAL(10,2),
                    gmv_2024 DECIMAL(10,2),
                    category_diversity_2024 INTEGER,
                    days_since_last_order INTEGER,
                    discount_rate_2024 DECIMAL(5,3),
                    refunds_count_2024 INTEGER,
                    refund_rate_2024 DECIMAL(5,3),
                    support_tickets_2024 INTEGER,
                    avg_csat_2024 DECIMAL(3,2),
                    emails_open_rate_90d DECIMAL(5,3),
                    emails_click_rate_90d DECIMAL(5,3),
                    review_count_2024 INTEGER,
                    avg_review_stars_2024 DECIMAL(3,2),
                    rfm_recency INTEGER,
                    rfm_frequency INTEGER,
                    rfm_monetary DECIMAL(10,2),
                    churn_label INTEGER,
                    _ingestion_timestamp TIMESTAMP,
                    _source_file VARCHAR(255),
                    _row_number INTEGER
                )
            """,
            
            "users_processed": """
                CREATE TABLE users_processed (
                    user_id VARCHAR(10) PRIMARY KEY,
                    age INTEGER,
                    country VARCHAR(50),
                    city VARCHAR(100),
                    reg_days INTEGER,
                    marketing_source VARCHAR(50),
                    sessions_30d INTEGER,
                    sessions_90d INTEGER,
                    avg_session_duration_90d DECIMAL(10,2),
                    median_pages_viewed_30d DECIMAL(10,2),
                    search_queries_30d INTEGER,
                    device_mix_ratio DECIMAL(5,3),
                    app_version_major VARCHAR(10),
                    orders_30d INTEGER,
                    orders_90d INTEGER,
                    orders_2024 INTEGER,
                    aov_2024 DECIMAL(10,2),
                    gmv_2024 DECIMAL(10,2),
                    category_diversity_2024 INTEGER,
                    days_since_last_order INTEGER,
                    discount_rate_2024 DECIMAL(5,3),
                    refunds_count_2024 INTEGER,
                    refund_rate_2024 DECIMAL(5,3),
                    support_tickets_2024 INTEGER,
                    avg_csat_2024 DECIMAL(3,2),
                    emails_open_rate_90d DECIMAL(5,3),
                    emails_click_rate_90d DECIMAL(5,3),
                    review_count_2024 INTEGER,
                    avg_review_stars_2024 DECIMAL(3,2),
                    rfm_recency INTEGER,
                    rfm_frequency INTEGER,
                    rfm_monetary DECIMAL(10,2),
                    churn_label INTEGER,
                    _processing_timestamp TIMESTAMP,
                    _processing_version VARCHAR(20)
                )
            """,
            
            "features": """
                CREATE TABLE features (
                    user_id VARCHAR(10) PRIMARY KEY,
                    rfm_segment VARCHAR(10),
                    rfm_score DECIMAL(5,2),
                    rfm_category VARCHAR(20),
                    session_intensity_30d DECIMAL(10,4),
                    session_intensity_90d DECIMAL(10,4),
                    engagement_ratio DECIMAL(10,4),
                    search_activity_ratio DECIMAL(10,4),
                    pages_per_session_30d DECIMAL(10,2),
                    email_engagement_score DECIMAL(5,3),
                    support_intensity DECIMAL(10,4),
                    reg_recency_category VARCHAR(20),
                    last_order_category VARCHAR(20),
                    order_frequency_2024 DECIMAL(10,4),
                    is_weekend_reg BOOLEAN,
                    is_month_end BOOLEAN,
                    value_per_session DECIMAL(10,2),
                    order_efficiency DECIMAL(10,4),
                    discount_sensitivity DECIMAL(10,2),
                    quality_score DECIMAL(5,2),
                    risk_score DECIMAL(10,4),
                    engagement_value DECIMAL(10,2),
                    clv_proxy DECIMAL(10,2),
                    purchase_consistency DECIMAL(10,4),
                    diversity_score DECIMAL(10,4),
                    is_latest_version BOOLEAN,
                    is_mobile_heavy BOOLEAN,
                    is_high_value BOOLEAN,
                    is_at_risk BOOLEAN,
                    _feature_engineering_timestamp TIMESTAMP
                )
            """,
            
            "model_predictions": """
                CREATE TABLE model_predictions (
                    user_id VARCHAR(10) PRIMARY KEY,
                    prediction DECIMAL(5,4),
                    prediction_class INTEGER,
                    confidence DECIMAL(5,4),
                    model_version VARCHAR(20),
                    prediction_timestamp TIMESTAMP,
                    features_used TEXT
                )
            """
        }
        
        self.create_tables(table_schemas)
        
        # Create indexes
        self._create_churn_indexes()
        # Create dimensional & fact tables for BI/star-schema
        self._create_dim_fact_tables()
    
    def _create_churn_indexes(self):
        # Tạo indexes cho các bảng churn prediction
        indexes = {
            "users_raw": [
                {"name": "idx_users_raw_churn", "columns": "churn_label"},
                {"name": "idx_users_raw_country", "columns": "country"},
                {"name": "idx_users_raw_marketing", "columns": "marketing_source"},
                {"name": "idx_users_raw_ingestion", "columns": "_ingestion_timestamp"}
            ],
            "users_processed": [
                {"name": "idx_users_processed_churn", "columns": "churn_label"},
                {"name": "idx_users_processed_country", "columns": "country"},
                {"name": "idx_users_processed_processing", "columns": "_processing_timestamp"}
            ],
            "features": [
                {"name": "idx_features_rfm", "columns": "rfm_segment"},
                {"name": "idx_features_high_value", "columns": "is_high_value"},
                {"name": "idx_features_at_risk", "columns": "is_at_risk"}
            ],
            "model_predictions": [
                {"name": "idx_predictions_class", "columns": "prediction_class"},
                {"name": "idx_predictions_confidence", "columns": "confidence"},
                {"name": "idx_predictions_timestamp", "columns": "prediction_timestamp"}
            ],
        }
        
        for table_name, table_indexes in indexes.items():
            self.create_indexes(table_name, table_indexes)

    def _create_dim_fact_tables(self):
        # Tạo lược đồ dimensional (star-schema) để hỗ trợ BI/analytics
        # Dimensions: dim_user, dim_date, dim_product, dim_channel, dim_device
        # Facts: fact_orders, fact_sessions
        table_schemas = {
            "dim_user": """
                CREATE TABLE dim_user (
                    user_key SERIAL PRIMARY KEY,
                    user_id VARCHAR(10) UNIQUE,
                    country VARCHAR(50),
                    city VARCHAR(100),
                    marketing_source VARCHAR(50),
                    reg_date DATE,
                    reg_days INTEGER,
                    cohort_month VARCHAR(7),
                    device_pref VARCHAR(20),
                    app_version_major VARCHAR(10),
                    is_high_value BOOLEAN,
                    is_at_risk BOOLEAN,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """,
            "dim_date": """
                CREATE TABLE dim_date (
                    date_key INTEGER PRIMARY KEY,
                    date_value DATE UNIQUE,
                    year INTEGER,
                    quarter INTEGER,
                    month INTEGER,
                    day INTEGER,
                    week INTEGER,
                    day_of_week INTEGER,
                    is_weekend BOOLEAN,
                    is_month_end BOOLEAN
                )
            """,
            "dim_product": """
                CREATE TABLE dim_product (
                    product_key SERIAL PRIMARY KEY,
                    product_id VARCHAR(50),
                    category VARCHAR(100),
                    subcategory VARCHAR(100),
                    brand VARCHAR(100),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """,
            "dim_channel": """
                CREATE TABLE dim_channel (
                    channel_key SERIAL PRIMARY KEY,
                    channel_code VARCHAR(50) UNIQUE,
                    channel_name VARCHAR(100),
                    channel_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """,
            "dim_device": """
                CREATE TABLE dim_device (
                    device_key SERIAL PRIMARY KEY,
                    device_type VARCHAR(20),
                    os VARCHAR(50),
                    app_version_major VARCHAR(10),
                    is_mobile_heavy BOOLEAN,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """,
            "fact_orders": """
                CREATE TABLE fact_orders (
                    order_id BIGSERIAL PRIMARY KEY,
                    user_key INTEGER,
                    date_key INTEGER,
                    product_key INTEGER,
                    channel_key INTEGER,
                    device_key INTEGER,
                    quantity INTEGER,
                    gmv DECIMAL(12,2),
                    aov DECIMAL(12,2),
                    discount_amount DECIMAL(12,2),
                    refund_amount DECIMAL(12,2),
                    is_refunded BOOLEAN,
                    country VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW(),
                    CONSTRAINT fk_order_user FOREIGN KEY(user_key) REFERENCES dim_user(user_key),
                    CONSTRAINT fk_order_date FOREIGN KEY(date_key) REFERENCES dim_date(date_key),
                    CONSTRAINT fk_order_product FOREIGN KEY(product_key) REFERENCES dim_product(product_key),
                    CONSTRAINT fk_order_channel FOREIGN KEY(channel_key) REFERENCES dim_channel(channel_key),
                    CONSTRAINT fk_order_device FOREIGN KEY(device_key) REFERENCES dim_device(device_key)
                )
            """,
            "fact_sessions": """
                CREATE TABLE fact_sessions (
                    session_id BIGSERIAL PRIMARY KEY,
                    user_key INTEGER,
                    date_key INTEGER,
                    device_key INTEGER,
                    channel_key INTEGER,
                    duration_seconds INTEGER,
                    pages_viewed INTEGER,
                    searches INTEGER,
                    conversions INTEGER,
                    gmv DECIMAL(12,2),
                    created_at TIMESTAMP DEFAULT NOW(),
                    CONSTRAINT fk_session_user FOREIGN KEY(user_key) REFERENCES dim_user(user_key),
                    CONSTRAINT fk_session_date FOREIGN KEY(date_key) REFERENCES dim_date(date_key),
                    CONSTRAINT fk_session_device FOREIGN KEY(device_key) REFERENCES dim_device(device_key),
                    CONSTRAINT fk_session_channel FOREIGN KEY(channel_key) REFERENCES dim_channel(channel_key)
                )
            """,
        }
        
        self.create_tables(table_schemas)

        indexes = {
            "dim_user": [
                {"name": "idx_dim_user_user_id", "columns": "user_id"},
                {"name": "idx_dim_user_country", "columns": "country"},
                {"name": "idx_dim_user_marketing_source", "columns": "marketing_source"}
            ],
            "dim_date": [
                {"name": "idx_dim_date_date_value", "columns": "date_value"}
            ],
            "dim_product": [
                {"name": "idx_dim_product_product_id", "columns": "product_id"},
                {"name": "idx_dim_product_category", "columns": "category"}
            ],
            "dim_channel": [
                {"name": "idx_dim_channel_code", "columns": "channel_code"}
            ],
            "dim_device": [
                {"name": "idx_dim_device_type", "columns": "device_type"}
            ],
            "fact_orders": [
                {"name": "idx_fact_orders_user", "columns": "user_key"},
                {"name": "idx_fact_orders_date", "columns": "date_key"},
                {"name": "idx_fact_orders_channel", "columns": "channel_key"},
                {"name": "idx_fact_orders_device", "columns": "device_key"}
            ],
            "fact_sessions": [
                {"name": "idx_fact_sessions_user", "columns": "user_key"},
                {"name": "idx_fact_sessions_date", "columns": "date_key"},
                {"name": "idx_fact_sessions_channel", "columns": "channel_key"},
                {"name": "idx_fact_sessions_device", "columns": "device_key"}
            ],
        }
        
        for table_name, table_indexes in indexes.items():
            self.create_indexes(table_name, table_indexes)


def main():
    # Hàm main để test data warehouse (CLI usage)
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Warehouse Operations")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--action", choices=["create", "info", "backup"], required=True, help="Action to perform")
    parser.add_argument("--table", help="Table name for info/backup actions")
    
    args = parser.parse_args()
    
    # Load configuration
    from utils.config import ConfigManager
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Initialize data warehouse
    dw = ChurnDataWarehouse(config)
    
    if args.action == "create":
        print("Tables created successfully")
    elif args.action == "info" and args.table:
        info = dw.get_table_info(args.table)
        print(json.dumps(info, indent=2, default=str))
    elif args.action == "backup" and args.table:
        backup_name = dw.backup_table(args.table)
        print(f"Backup created: {backup_name}")


if __name__ == "__main__":
    main()
