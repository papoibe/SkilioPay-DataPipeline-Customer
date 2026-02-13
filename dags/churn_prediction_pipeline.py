
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
import os
import sys
import pandas as pd  # Thêm import pandas cho load_to_warehouse function

# Thêm src vào path để import các module từ project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingestion.csv_ingestion import CSVIngestion
from processing.etl_pipeline import ETLPipeline
from storage.data_warehouse import ChurnDataWarehouse
from ml.model_trainer import ModelTrainer
from utils.config import ConfigManager
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Default arguments cho DAG
# Cấu hình mặc định cho tất cả tasks trong pipeline
default_args = {
    'owner': 'Le Nguyen Phuoc Thinh',  # Người sở hữu pipeline
    'depends_on_past': False,  # Task không phụ thuộc vào lần chạy trước
    'start_date': datetime(2024, 1, 1),  # Ngày bắt đầu chạy DAG
    'email_on_failure': True,  # Gửi email khi task thất bại
    'email_on_retry': False,  # Không gửi email khi retry
    'retries': 2,  # Số lần retry khi task thất bại
    'retry_delay': timedelta(minutes=5),  # Thời gian chờ giữa các lần retry
    'email': ['lenguyenphuocthinh1234@gmail.com']  # Email nhận thông báo
}

# DAG definition - Định nghĩa workflow cho churn prediction pipeline
# DAG này chạy hàng ngày, tự động orchestrate toàn bộ quy trình từ ingestion đến model deployment
dag = DAG(
    'skilio_pay_churn_prediction_pipeline',
    default_args=default_args,
    description='Complete churn prediction pipeline for SkilioPay',
    schedule_interval='@daily',  # Chạy hàng ngày
    catchup=False,  # Không chạy lại các lần đã bỏ lỡ
    max_active_runs=1,  # Chỉ cho phép 1 lần chạy đồng thời
    tags=['churn', 'prediction', 'ml', 'etl']  # Tags để phân loại trong Airflow UI
)


def load_config():
    # Load configuration từ config.yaml
    # Trả về dictionary chứa toàn bộ cấu hình cho pipeline
    config_manager = ConfigManager()
    return config_manager.load_config()


def check_data_quality(**context):
    # Kiểm tra chất lượng dữ liệu trước khi xử lý
    # Kiểm tra file input có tồn tại và có kích thước hợp lệ không
    config = context['ti'].xcom_pull(task_ids='load_config')
    
    # Kiểm tra file input có tồn tại không
    input_file = config['data_sources']['csv']['input_path']
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Kiểm tra kích thước file (phải >= 1KB)
    file_size = os.path.getsize(input_file)
    if file_size < 1000:  # Nhỏ hơn 1KB
        raise ValueError(f"Input file too small: {file_size} bytes")
    
    logger.info(f"Data quality check passed. File size: {file_size} bytes")
    return {"file_size": file_size, "status": "passed"}


def ingest_data(**context):
    # Ingest raw data từ CSV file
    # Đọc CSV, validate theo schema, và convert sang Parquet format
    config = context['ti'].xcom_pull(task_ids='load_config')
    
    ingestion = CSVIngestion(config)
    
    input_path = config['data_sources']['csv']['input_path']
    output_path = f"data/raw/churn_data_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    metadata = ingestion.ingest_csv(
        input_path=input_path,
        output_path=output_path,
        schema_file="config/schemas/churn_schema.json",
        validate_data=True
    )
    
    logger.info(f"Data ingestion completed: {metadata}")
    return metadata


def process_data(**context):
    # Xử lý và làm sạch dữ liệu
    # Chạy ETL pipeline: cleaning, validation, feature engineering
    config = context['ti'].xcom_pull(task_ids='load_config')
    ingestion_metadata = context['ti'].xcom_pull(task_ids='ingest_data')
    
    etl = ETLPipeline(config)
    
    input_path = ingestion_metadata['output_file']
    output_path = f"data/processed/churn_processed_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    metadata = etl.run_pipeline(
        input_path=input_path,
        output_path=output_path,
        schema_file="config/schemas/churn_schema.json",
        run_quality_checks=True,
        run_feature_engineering=True
    )
    
    logger.info(f"Data processing completed: {metadata}")
    return metadata


def load_to_warehouse(**context):
    # Load processed data vào data warehouse (PostgreSQL)
    # Load vào các bảng: users_processed và features
    config = context['ti'].xcom_pull(task_ids='load_config')
    processing_metadata = context['ti'].xcom_pull(task_ids='process_data')
    
    # Khởi tạo kết nối data warehouse
    dw = ChurnDataWarehouse(config, create_tables=False)  # Giả định bảng đã tồn tại
    
    # Đọc processed data từ Parquet file
    df = pd.read_parquet(processing_metadata['output_file'])
    
    # Load vào bảng users_processed (dữ liệu đã xử lý)
    dw.load_data(df, "users_processed", if_exists="replace")
    
    # Load features vào bảng features nếu có
    # Lọc bỏ các cột metadata (bắt đầu bằng _) và chỉ giữ features
    if 'features' in df.columns:
        features_df = df[['user_id'] + [col for col in df.columns if col.startswith('_') == False and col != 'user_id']]
        dw.load_data(features_df, "features", if_exists="replace")
    
    logger.info("Data loaded to warehouse successfully")
    return {"status": "success", "rows_loaded": len(df)}


def train_model(**context):
    # Huấn luyện mô hình ML cho churn prediction
    # Chuẩn bị dữ liệu, train model, và lưu kết quả
    config = context['ti'].xcom_pull(task_ids='load_config')
    processing_metadata = context['ti'].xcom_pull(task_ids='process_data')
    
    # Load processed data
    df = pd.read_parquet(processing_metadata['output_file'])
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Prepare data
    X, y, feature_columns = trainer.prepare_data(df)
    
    # Train model
    results = trainer.train_model(X, y)
    
    logger.info(f"Model training completed: {results}")
    return results


def evaluate_model(**context):
    # Đánh giá hiệu suất mô hình
    # Kiểm tra metrics (accuracy, ROC-AUC) có đạt ngưỡng tối thiểu không
    training_results = context['ti'].xcom_pull(task_ids='train_model')
    
    # Lấy test metrics từ kết quả training
    test_metrics = training_results['test_metrics']
    
    # Kiểm tra model có đạt ngưỡng hiệu suất tối thiểu không
    # Accuracy >= 0.75 và ROC-AUC >= 0.80
    min_accuracy = 0.75
    min_roc_auc = 0.80
    
    if test_metrics['accuracy'] < min_accuracy:
        raise ValueError(f"Model accuracy {test_metrics['accuracy']} below minimum {min_accuracy}")
    
    if test_metrics['roc_auc'] < min_roc_auc:
        raise ValueError(f"Model ROC AUC {test_metrics['roc_auc']} below minimum {min_roc_auc}")
    
    logger.info(f"Model evaluation passed: {test_metrics}")
    return {"status": "passed", "metrics": test_metrics}


def deploy_model(**context):
    # Deploy model cho serving
    # Copy model file đến thư mục serving để API có thể sử dụng
    training_results = context['ti'].xcom_pull(task_ids='train_model')
    
    # Copy model file đến thư mục serving
    # API sẽ load model từ đây để phục vụ predictions
    model_path = training_results['model_path']
    serving_path = "models/churn_model_latest.joblib"
    
    import shutil
    shutil.copy2(model_path, serving_path)  # Copy và giữ nguyên metadata (timestamp, permissions)
    
    logger.info(f"Model deployed to {serving_path}")
    return {"status": "deployed", "model_path": serving_path}


def send_success_notification(**context):
    # Gửi thông báo thành công (được xử lý bởi EmailOperator)
    return "Pipeline completed successfully"


def send_failure_notification(**context):
    # Gửi thông báo thất bại (được xử lý bởi EmailOperator)
    return "Pipeline failed - check logs for details"


# Task definitions - Định nghĩa các tasks trong DAG
# Task 1: Load configuration
load_config_task = PythonOperator(
    task_id='load_config',
    python_callable=load_config,
    dag=dag
)

# Task 2: Kiểm tra chất lượng dữ liệu
check_data_quality_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)

# Task 3: Ingest raw data từ CSV
ingest_data_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag
)

# Task 4: Xử lý và làm sạch dữ liệu
process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag
)

# Task 5: Load processed data vào data warehouse
load_to_warehouse_task = PythonOperator(
    task_id='load_to_warehouse',
    python_callable=load_to_warehouse,
    dag=dag
)

# Task 6: Huấn luyện mô hình ML
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

# Task 7: Đánh giá hiệu suất mô hình
evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

# Task 8: Deploy model cho serving
deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Success notification - Gửi email khi pipeline thành công
success_notification = EmailOperator(
    task_id='send_success_notification',
    to=['lenguyenphuocthinh1234@gmail.com'],
    subject='SkilioPay Pipeline Success',
    html_content='<p>Churn prediction pipeline completed successfully.</p>',
    dag=dag,
    trigger_rule='all_success'  # Chỉ trigger khi tất cả upstream tasks thành công
)

# Failure notification - Gửi email khi pipeline thất bại
failure_notification = EmailOperator(
    task_id='send_failure_notification',
    to=['lenguyenphuocthinh1234@gmail.com'],
    subject='SkilioPay Pipeline Failure',
    html_content='<p>Churn prediction pipeline failed. Please check logs.</p>',
    dag=dag,
    trigger_rule='one_failed'  # Trigger khi có ít nhất 1 upstream task thất bại
)

# Task dependencies - Định nghĩa thứ tự thực thi các tasks
# Flow: load_config → check_data_quality → ingest_data → process_data
load_config_task >> check_data_quality_task >> ingest_data_task >> process_data_task

# Sau khi process_data xong, chạy song song: load_to_warehouse và train_model
process_data_task >> [load_to_warehouse_task, train_model_task]

# Flow training: train_model → evaluate_model → deploy_model
train_model_task >> evaluate_model_task >> deploy_model_task

# Sau khi cả load_to_warehouse và deploy_model xong, gửi notifications
[load_to_warehouse_task, deploy_model_task] >> success_notification
[load_to_warehouse_task, deploy_model_task] >> failure_notification
