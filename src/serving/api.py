# FastAPI 
# Cung cấp REST API endpoints cho:
# - Real-time churn prediction
# - Batch prediction
# - Model information
# - Health checks
# - Data access
#
# DATA FLOW TRONG PRODUCTION:
# 1. Training Phase (Offline):
#    CSV file → ETL Pipeline → Data Warehouse → Train Model → Save Model
#
# 2. Serving Phase (Online/Real-time):
#    Option A: Client gửi full data
#    Client → POST /predict {full_user_data} → API → Predict → Response
#    Option B: API tự lấy data từ warehouse (RECOMMENDED)
#    Client → GET /predict/{user_id} → API query warehouse → Predict → Response
#    Data Warehouse Sources (Production):
#    - PostgreSQL: Data đã được ETL từ CSV/streams
#    - Event Streams: Real-time user actions (Kafka, RabbitMQ)
#    - Microservices: Aggregated data từ User Service, Order Service
#    - Data Lake: Latest snapshots (S3, HDFS)
#

# 1. Load CSV vào warehouse: Chạy ETL pipeline để load CSV → PostgreSQL
# 2. Test API: Dùng GET /predict/{user_id} - API tự động lấy data từ warehouse
# 3. Hoặc: Dùng POST /predict với data từ CSV (manual)
# Ví dụ test:
# - Load data: python -m src.processing.etl_pipeline (load vào warehouse)
# - Test API: curl http://localhost:8000/predict/USER001

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import json
from pathlib import Path

from utils.logging_config import get_logger
from utils.config import ConfigManager
# from ml.model_trainer import ModelTrainer  # Không cần import nếu không dùng trong API
from utils.cache_manager import PipelineCache
from storage.data_warehouse import ChurnDataWarehouse

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SkilioPay Churn Prediction API",
    description="API for churn prediction and data serving",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables

model = None
scaler = None
feature_columns = []
config = None
data_warehouse = None  # Data warehouse connection để lấy data real-time
prediction_cache = PipelineCache(cache_dir="cache/predict", default_ttl=3600)


class PredictionRequest(BaseModel):
    # Request model cho single prediction - chứa tất cả thông tin user cần thiết để predict churn
    user_id: str = Field(..., description="User ID")
    age: int = Field(..., ge=13, le=100, description="User age")
    country: str = Field(..., description="User country")
    city: str = Field(..., description="User city")
    reg_days: int = Field(..., ge=0, description="Days since registration")
    marketing_source: str = Field(..., description="Marketing source")
    sessions_30d: int = Field(..., ge=0, description="Sessions in last 30 days")
    sessions_90d: int = Field(..., ge=0, description="Sessions in last 90 days")
    avg_session_duration_90d: float = Field(..., ge=0, description="Average session duration")
    median_pages_viewed_30d: float = Field(..., ge=0, description="Median pages viewed")
    search_queries_30d: int = Field(..., ge=0, description="Search queries in 30 days")
    device_mix_ratio: float = Field(..., ge=0, le=1, description="Device mix ratio")
    app_version_major: str = Field(..., description="App version major")
    orders_30d: int = Field(..., ge=0, description="Orders in last 30 days")
    orders_90d: int = Field(..., ge=0, description="Orders in last 90 days")
    orders_2024: int = Field(..., ge=0, description="Orders in 2024")
    aov_2024: float = Field(..., ge=0, description="Average order value 2024")
    gmv_2024: float = Field(..., ge=0, description="Gross merchandise value 2024")
    category_diversity_2024: int = Field(..., ge=0, description="Category diversity 2024")
    days_since_last_order: int = Field(..., ge=0, le=365, description="Days since last order")
    discount_rate_2024: float = Field(..., ge=0, le=1, description="Discount rate 2024")
    refunds_count_2024: int = Field(..., ge=0, description="Refunds count 2024")
    refund_rate_2024: float = Field(..., ge=0, le=1, description="Refund rate 2024")
    support_tickets_2024: int = Field(..., ge=0, description="Support tickets 2024")
    avg_csat_2024: float = Field(..., ge=1, le=5, description="Average CSAT 2024")
    emails_open_rate_90d: float = Field(..., ge=0, le=1, description="Email open rate 90d")
    emails_click_rate_90d: float = Field(..., ge=0, le=1, description="Email click rate 90d")
    review_count_2024: int = Field(..., ge=0, description="Review count 2024")
    avg_review_stars_2024: float = Field(..., ge=1, le=5, description="Average review stars 2024")
    rfm_recency: int = Field(..., ge=0, le=365, description="RFM recency")
    rfm_frequency: int = Field(..., ge=0, description="RFM frequency")
    rfm_monetary: float = Field(..., ge=0, description="RFM monetary")


class BatchPredictionRequest(BaseModel):
    # Request model cho batch prediction - nhận danh sách nhiều users để predict cùng lúc
    data: List[Dict[str, Any]] = Field(..., description="List of user data for prediction")


class PredictionResponse(BaseModel):
    # Response model cho prediction - trả về kết quả dự đoán churn với probability và confidence
    user_id: str
    prediction: float = Field(..., ge=0, le=1, description="Churn probability")
    prediction_class: int = Field(..., ge=0, le=1, description="Predicted class (0=no churn, 1=churn)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_version: str
    prediction_timestamp: str


class HealthResponse(BaseModel):
    # Response model cho health check - kiểm tra trạng thái API và model
    status: str
    timestamp: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    # Response model cho model information - thông tin về mô hình đang sử dụng
    algorithm: str
    feature_count: int
    feature_columns: List[str]
    model_version: str
    training_date: str
    performance_metrics: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    # Khởi tạo application khi API khởi động
    # Load config, kết nối data warehouse, load model từ disk
    global model, scaler, feature_columns, config, data_warehouse
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config("config/config.yaml")
        
        # Initialize data warehouse connection để lấy data real-time
        # Trong production: Data warehouse chứa data mới nhất từ các nguồn (DB, streams, APIs)
        # Với CSV hiện tại: Load CSV vào warehouse trước, sau đó query từ đây
        try:
            data_warehouse = ChurnDataWarehouse(config, create_tables=False)
            logger.info("Data warehouse connection initialized")
        except Exception as e:
            logger.warning(f"Data warehouse connection failed: {str(e)}. API will work with manual data input only.")
            data_warehouse = None
        
        # Load model
        # Find latest model file if specific path varies
        model_dir = Path("models")
        model_files = list(model_dir.glob("churn_model_*.joblib"))
        
        if model_files:
            # Sort by modification time (newest first)
            latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading latest model: {latest_model_path}")
            
            model_data = joblib.load(latest_model_path)
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            logger.info(f"Model loaded successfully. Features: {len(feature_columns)}")
        else:
            logger.warning("No trained model found in models/. Please train a model first.")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    # Health check endpoint - kiểm tra API có đang hoạt động và model có được load không
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    # Lấy thông tin về mô hình đang sử dụng (algorithm, features, version, metrics)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        algorithm=type(model).__name__,
        feature_count=len(feature_columns),
        feature_columns=feature_columns,
        model_version="1.0.0",
        training_date=datetime.now().isoformat(),
        performance_metrics={}  # You would load actual metrics here
    )


@app.get("/predict/{user_id}", response_model=PredictionResponse)
async def predict_churn_by_id(user_id: str):
    # Predict churn chỉ với user_id - tự động lấy data từ warehouse
    # Đây là cách đơn giản nhất để test API:
    # 1. Load CSV vào warehouse trước (chạy ETL pipeline)
    # 2. Gọi endpoint này với user_id
    # 3. API tự động lấy data từ warehouse và predict
    # Ví dụ: GET /predict/USER001
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if data_warehouse is None:
        raise HTTPException(
            status_code=503, 
            detail="Data warehouse not available. Use /predict endpoint with full data instead."
        )
    
    try:
        # Lấy user data từ warehouse (gọi trực tiếp logic, không dùng await vì đã là async)
        if data_warehouse is None:
            raise HTTPException(
                status_code=503, 
                detail="Data warehouse not available. Use /predict endpoint with full data instead."
            )
        
        # Query user data từ warehouse
        query = f"""
        SELECT * FROM {data_warehouse.schema}.features 
        WHERE user_id = :user_id
        """
        df = data_warehouse.query_data(query, params={"user_id": user_id})
        
        if df.empty:
            query = f"""
            SELECT * FROM {data_warehouse.schema}.users_processed 
            WHERE user_id = :user_id
            """
            df = data_warehouse.query_data(query, params={"user_id": user_id})
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found in warehouse")
        
        user_data = df.iloc[0].to_dict()
        
        # Tạo request từ warehouse data
        # Loại bỏ các cột không cần thiết (metadata, target, etc.)
        exclude_cols = ['_feature_engineering_timestamp', 'churn_label', '_processing_timestamp']
        user_data_clean = {k: v for k, v in user_data.items() if k not in exclude_cols and not k.startswith('_')}
        
        # Tạo DataFrame
        df = pd.DataFrame([user_data_clean])
        
        # Prepare features - chỉ lấy các cột có trong feature_columns
        # Đảm bảo chỉ lấy numerical columns (categorical đã được one-hot encoded khi train)
        available_cols = [col for col in feature_columns if col in df.columns]
        X = df[available_cols].copy()
        
        # Chỉ lấy numerical columns (categorical đã được encode khi train)
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numerical_cols].copy()
        
        # Handle missing values - chỉ cho numerical
        if len(X.columns) > 0:
            X = X.fillna(X.median())
        
        # Scale features - chỉ scale numerical columns
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction_proba = model.predict_proba(X_scaled)[0]
        prediction_class = model.predict(X_scaled)[0]
        confidence = max(prediction_proba)
        
        response = PredictionResponse(
            user_id=user_id,
            prediction=float(prediction_proba[1]),
            prediction_class=int(prediction_class),
            confidence=float(confidence),
            model_version="1.0.0",
            prediction_timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    # Predict churn với full user data trong request body
    # Cách 1: Gửi đầy đủ data trong request (như hiện tại)
    # Cách 2: Dùng /predict/{user_id} để tự động lấy từ warehouse
    # Trong production: 
    # - Client có thể gửi data mới nhất từ hệ thống của họ
    # - Hoặc chỉ gửi user_id, API tự query từ warehouse
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Cache theo nội dung request (model_dump thay cho dict, tương thích Pydantic V2)
        cached = prediction_cache.get(operation="predict_single", data=request.model_dump())
        if cached is not None:
            return cached
        # Convert request to DataFrame
        user_data = request.model_dump()
        user_id = user_data.pop('user_id')
        
        # Create DataFrame with single row
        df = pd.DataFrame([user_data])
        
        # Prepare features (you would implement feature engineering here)
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        
        # Encode categorical features (simplified)
        categorical_cols = config.get("ml", {}).get("features", {}).get("categorical", [])
        for col in categorical_cols:
            if col in X.columns and X[col].dtype == 'object':
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction_proba = model.predict_proba(X_scaled)[0]
        prediction_class = model.predict(X_scaled)[0]
        confidence = max(prediction_proba)
        response = PredictionResponse(
            user_id=user_id,
            prediction=float(prediction_proba[1]),  # Probability of churn
            prediction_class=int(prediction_class),
            confidence=float(confidence),
            model_version="1.0.0",
            prediction_timestamp=datetime.now().isoformat()
        )
        # Lưu cache
        prediction_cache.set(operation="predict_single", value=response, data=request.dict())
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict_churn(request: BatchPredictionRequest):
    # Predict churn cho nhiều users cùng lúc (batch prediction)
    # Nhận danh sách users và trả về predictions cho tất cả
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        if 'user_id' not in df.columns:
            raise HTTPException(status_code=400, detail="user_id column is required")
        
        user_ids = df['user_id'].tolist()
        
        # Prepare features
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        
        # Encode categorical features
        categorical_cols = config.get("ml", {}).get("features", {}).get("categorical", [])
        for col in categorical_cols:
            if col in X.columns and X[col].dtype == 'object':
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions (có thể cân nhắc cache theo từng user nếu cần)
        predictions_proba = model.predict_proba(X_scaled)
        predictions_class = model.predict(X_scaled)
        
        # Prepare response
        results = []
        for i, user_id in enumerate(user_ids):
            results.append({
                "user_id": user_id,
                "prediction": float(predictions_proba[i][1]),
                "prediction_class": int(predictions_class[i]),
                "confidence": float(max(predictions_proba[i])),
                "model_version": "1.0.0",
                "prediction_timestamp": datetime.now().isoformat()
            })
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/data/users/{user_id}")
async def get_user_data(user_id: str):
    # Lấy data của user từ data warehouse (real-time data source)
    # Trong production: Data này đến từ:
    # - PostgreSQL warehouse (đã load từ CSV hoặc real-time streams)
    # - Event streams (Kafka, RabbitMQ) - user actions mới nhất
    # - Microservices APIs - aggregated data từ nhiều services
    # Với CSV hiện tại: 
    # - Bước 1: Load CSV vào warehouse bằng ETL pipeline
    # - Bước 2: Query từ warehouse ở đây
    if data_warehouse is None:
        raise HTTPException(
            status_code=503, 
            detail="Data warehouse not available. Please load data into warehouse first."
        )
    
    try:
        # Query user data từ warehouse (từ bảng users_processed hoặc features)
        # Ưu tiên lấy từ features vì đã có feature engineering sẵn
        query = f"""
        SELECT * FROM {data_warehouse.schema}.features 
        WHERE user_id = :user_id
        """
        df = data_warehouse.query_data(query, params={"user_id": user_id})
        
        if df.empty:
            # Nếu không có trong features, thử lấy từ users_processed
            query = f"""
            SELECT * FROM {data_warehouse.schema}.users_processed 
            WHERE user_id = :user_id
            """
            df = data_warehouse.query_data(query, params={"user_id": user_id})
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found in warehouse")
        
        # Convert to dict
        user_data = df.iloc[0].to_dict()
        return {"user_id": user_id, "data": user_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user data: {str(e)}")


@app.get("/data/users")
async def get_users(limit: int = 100, offset: int = 0):
    # Lấy danh sách users từ data warehouse
    # Dùng để test hoặc lấy sample data
    # Parameters: limit (số users tối đa), offset (bắt đầu từ user thứ mấy)
    if data_warehouse is None:
        return {
            "message": "Data warehouse not available. Please load data into warehouse first.",
            "limit": limit,
            "offset": offset,
            "users": []
        }
    
    try:
        query = f"""
        SELECT user_id FROM {data_warehouse.schema}.features 
        LIMIT :limit OFFSET :offset
        """
        df = data_warehouse.query_data(query, params={"limit": limit, "offset": offset})
        user_ids = df['user_id'].tolist() if not df.empty else []
        
        return {
            "users": user_ids,
            "count": len(user_ids),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to get users: {str(e)}")
        return {"users": [], "count": 0, "limit": limit, "offset": offset}


@app.get("/metrics")
async def get_metrics():
    # Lấy application metrics (tổng số predictions, model accuracy, uptime)
    return {
        "total_predictions": 0,  # You would track this
        "model_accuracy": 0.85,  # You would load actual metrics
        "uptime": "24h",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("config/config.yaml")
    
    api_config = config.get("api", {})
    
    # Sửa path để uvicorn tìm đúng app
    uvicorn.run(
        "src.serving.api:app",  # Đúng path khi chạy từ root
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("reload", False),  # Tắt reload để tránh lỗi
        log_level=api_config.get("log_level", "info")
    )
