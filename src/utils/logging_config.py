# thiết lập nhật ký cho cả quy trình
# dùng cho ingestion, ETL, ML, API, DAG
import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime
import structlog

# Hàm thiết lập logging
def setup_logging(config: Dict[str, Any] = None) -> None:
    if config is None:
        config = {
            "level": "INFO",
            "format": "json",
            "file": "logs/pipeline.log"
        }
    
    # Create logs directory
    log_file = config.get("file", "logs/pipeline.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    log_level = getattr(logging, config.get("level", "INFO").upper())
    # Lấy format từ config, nếu không có thì sử dụng json
    log_format = config.get("format", "json")
    
    if log_format == "json":
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False)
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    # Trả về logger dạng structured (structlog) theo tên "name"
    return structlog.get_logger(name)


class PipelineLogger:
    # Logger tuỳ biến cho các hoạt động của pipeline (ingestion, processing, training, ...)
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_ingestion_start(self, source: str, file_path: str):
        # Ghi log bắt đầu bước ingestion (nguồn và đường dẫn file)
        self.logger.info(
            "Data ingestion started",
            source=source,
            file_path=file_path,
            operation="ingestion_start"
        )
    
    def log_ingestion_complete(self, source: str, rows: int, duration: float):
        # Ghi log hoàn thành ingestion (số dòng và thời gian)
        self.logger.info(
            "Data ingestion completed",
            source=source,
            rows=rows,
            duration_seconds=duration,
            operation="ingestion_complete"
        )
    
    def log_processing_start(self, stage: str, rows: int):
        # Ghi log bắt đầu xử lý dữ liệu (tên stage và số dòng đầu vào)
        self.logger.info(
            "Data processing started",
            stage=stage,
            input_rows=rows,
            operation="processing_start"
        )
    
    def log_processing_complete(self, stage: str, input_rows: int, output_rows: int, duration: float):
        # Ghi log hoàn thành xử lý (số dòng vào/ra và thời gian)
        self.logger.info(
            "Data processing completed",
            stage=stage,
            input_rows=input_rows,
            output_rows=output_rows,
            duration_seconds=duration,
            operation="processing_complete"
        )
    
    def log_model_training_start(self, algorithm: str, features: int, samples: int):
        # Ghi log bắt đầu huấn luyện model (thuật toán, số feature, số mẫu)
        self.logger.info(
            "Model training started",
            algorithm=algorithm,
            feature_count=features,
            sample_count=samples,
            operation="model_training_start"
        )
    
    def log_model_training_complete(self, algorithm: str, metrics: Dict[str, float], duration: float):
        # Ghi log hoàn thành huấn luyện (thuật toán, metrics, thời gian)
        self.logger.info(
            "Model training completed",
            algorithm=algorithm,
            metrics=metrics,
            duration_seconds=duration,
            operation="model_training_complete"
        )
    
    def log_prediction_request(self, user_id: str, prediction: float, confidence: float):
        # Ghi log yêu cầu dự đoán (user, xác suất và độ tự tin)
        self.logger.info(
            "Prediction requested",
            user_id=user_id,
            prediction=prediction,
            confidence=confidence,
            operation="prediction_request"
        )
    
    def log_error(self, operation: str, error: str, context: Dict[str, Any] = None):
        # Ghi log lỗi kèm bối cảnh (operation, thông tin lỗi, ngữ cảnh)
        self.logger.error(
            "Operation failed",
            operation=operation,
            error=error,
            context=context or {},
            operation_type="error"
        )
    
    def log_data_quality_issue(self, check_name: str, issue: str, affected_rows: int):
        # Ghi log vấn đề chất lượng dữ liệu (tên check, mô tả, số dòng ảnh hưởng)
        self.logger.warning(
            "Data quality issue detected",
            check_name=check_name,
            issue=issue,
            affected_rows=affected_rows,
            operation="data_quality_issue"
        )

# Logger chuyên cho metrics (gía trị, counter, timing)
class MetricsLogger:
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        # Ghi log một metric với tên, giá trị và thẻ (tags)
        self.logger.info(
            "Metric recorded",
            metric_name=metric_name,
            value=value,
            tags=tags or {},
            operation="metric_log"
        )
    
    def log_counter(self, counter_name: str, increment: int = 1, tags: Dict[str, str] = None):
        # Ghi log tăng counter với số bước tăng và thẻ (tags)
        self.logger.info(
            "Counter incremented",
            counter_name=counter_name,
            increment=increment,
            tags=tags or {},
            operation="counter_log"
        )
    
    def log_timing(self, operation: str, duration: float, tags: Dict[str, str] = None):
        # Ghi log thời gian thực thi cho một operation
        self.logger.info(
            "Timing recorded",
            operation=operation,
            duration_seconds=duration,
            tags=tags or {},
            # operation="timing_log"
            time="timing_log"
        )


def setup_mlflow_logging():
    # Thiết lập tích hợp ghi log với MLflow (tracking URI, experiment, autolog)
    import mlflow
    
    # Configure MLflow logging
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn_prediction")
    
    # Enable automatic logging
    mlflow.sklearn.autolog()


def create_logging_config() -> Dict[str, Any]:
    # Tạo dictionary cấu hình logging (formatters, handlers, root logger)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "json": {
                "()": "structlog.stdlib.ProcessorFormatter",
                "processor": "structlog.dev.ConsoleRenderer"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "json",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": "logs/pipeline.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }


def main():
    # Hàm main để test nhanh cấu hình logging (console/file) và các logger tiện ích
    import argparse
    
    parser = argparse.ArgumentParser(description="Logging Configuration")
    parser.add_argument("--config", help="Logging configuration file")
    parser.add_argument("--test", action="store_true", help="Test logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        setup_logging(config)
    else:
        setup_logging()
    
    if args.test:
        # Test logging
        logger = get_logger("test")
        logger.info("Test log message", test=True)
        
        pipeline_logger = PipelineLogger("test_pipeline")
        pipeline_logger.log_ingestion_start("test_source", "test_file.csv")
        pipeline_logger.log_ingestion_complete("test_source", 1000, 5.5)
        
        metrics_logger = MetricsLogger("test_metrics")
        metrics_logger.log_metric("test_metric", 42.5, {"tag1": "value1"})
        metrics_logger.log_counter("test_counter", 1)
        metrics_logger.log_timing("test_operation", 1.23)


if __name__ == "__main__":
    main()
