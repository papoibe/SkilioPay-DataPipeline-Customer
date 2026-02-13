-- init.sql
-- Script khởi tạo database cho Docker Compose
-- Tạo database chính, database Airflow, schema và bảng cần thiết

-- Tạo database cho Airflow (PostgreSQL container mặc định tạo skilio_pay)
CREATE DATABASE airflow;

-- Tạo schema cho churn prediction
CREATE SCHEMA IF NOT EXISTS churn_prediction;

-- Tạo bảng users_raw (Bronze layer - dữ liệu thô từ ingestion)
CREATE TABLE IF NOT EXISTS churn_prediction.users_raw (
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
);

-- Tạo bảng users_processed (Silver layer - dữ liệu đã xử lý)
CREATE TABLE IF NOT EXISTS churn_prediction.users_processed (
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
);

-- Tạo bảng features (Gold layer - feature store)
CREATE TABLE IF NOT EXISTS churn_prediction.features (
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
);

-- Tạo bảng model_predictions (kết quả dự đoán)
CREATE TABLE IF NOT EXISTS churn_prediction.model_predictions (
    prediction_id SERIAL PRIMARY KEY,
    user_id VARCHAR(10),
    churn_probability DECIMAL(5,4),
    churn_prediction INTEGER,
    model_version VARCHAR(50),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence DECIMAL(5,4)
);

-- Tạo indexes để tối ưu query performance
CREATE INDEX IF NOT EXISTS idx_users_raw_country ON churn_prediction.users_raw(country);
CREATE INDEX IF NOT EXISTS idx_users_processed_country ON churn_prediction.users_processed(country);
CREATE INDEX IF NOT EXISTS idx_users_processed_churn ON churn_prediction.users_processed(churn_label);
CREATE INDEX IF NOT EXISTS idx_features_user ON churn_prediction.features(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON churn_prediction.model_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON churn_prediction.model_predictions(prediction_timestamp);
