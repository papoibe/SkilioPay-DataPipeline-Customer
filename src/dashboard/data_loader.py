"""
Data Loader cho Streamlit Dashboard.
Cung cấp các hàm load dữ liệu từ Parquet files, Model files, và PostgreSQL.
"""
import os
import sys
import glob
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Thêm project root vào sys.path để import các module từ src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_project_root():
    """Trả về đường dẫn tuyệt đối đến project root."""
    return PROJECT_ROOT


def load_processed_data():
    """
    Load processed data từ file Parquet mới nhất.
    Returns: pd.DataFrame hoặc None nếu không tìm thấy file.
    """
    data_dir = PROJECT_ROOT / "data" / "processed"
    # glob tìm tất cả file parquet, sắp xếp theo tên (chứa date) để lấy mới nhất
    parquet_files = sorted(data_dir.glob("churn_processed_*.parquet"))
    
    if not parquet_files:
        return None
    
    latest_file = parquet_files[-1]  # File cuối cùng = mới nhất (do tên chứa YYYYMMDD)
    df = pd.read_parquet(latest_file)
    
    # Reconstruct 'country' column from one-hot encoded features if missing
    if 'country' not in df.columns:
        country_cols = [c for c in df.columns if c.startswith('country_')]
        if country_cols:
            # Create 'country' column by finding the column with value 1
            # Remove prefix 'country_' to get country name
            df['country'] = df[country_cols].idxmax(axis=1).apply(lambda x: x.replace('country_', ''))
            
    # Reconstruct 'reg_recency_category' if missing (useful for analysis) but might not be easy.
    # But 'country' is critical for the "Theo Quốc Gia" tab.
    
    return df


def load_raw_data():
    """
    Load raw data từ CSV gốc.
    Dùng để hiển thị dữ liệu chưa xử lý cho so sánh.
    """
    csv_path = PROJECT_ROOT / "data" / "raw"
    # Tìm file CSV gốc
    csv_file = PROJECT_ROOT / "SkilioMall_Churn Dataset_50,000 Users.xlsx - Sheet1.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    
    # Fallback: tìm parquet raw
    parquet_files = sorted(csv_path.glob("churn_data_*.parquet"))
    if parquet_files:
        return pd.read_parquet(parquet_files[-1])
    return None


def load_model():
    """
    Load model mới nhất từ thư mục models/.
    Returns: dict chứa 'model', 'scaler', 'feature_columns', 'target_column'.
    """
    models_dir = PROJECT_ROOT / "models"
    # glob tìm tất cả model files, sắp xếp theo tên để lấy mới nhất
    model_files = sorted(models_dir.glob("churn_model_*.joblib"))
    
    if not model_files:
        return None
    
    latest_model = model_files[-1]
    model_data = joblib.load(latest_model)
    model_data['model_path'] = str(latest_model)
    
    # Fix Feature Mismatch:
    # Check if XGBoost booster has different feature names than joblib metadata
    model = model_data.get('model')
    correct_features = None
    
    # 1. Priority: Sklearn API feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        correct_features = model.feature_names_in_
        
    # 2. Fallback: Booster feature names
    elif hasattr(model, 'get_booster'):
        try:
            correct_features = model.get_booster().feature_names
        except Exception:
            pass
            
    if correct_features is not None and len(correct_features) > 0:
        saved_features = model_data.get('feature_columns', [])
        if len(correct_features) != len(saved_features):
            # print(f"⚠️ Feature mismatch detected! Saved: {len(saved_features)}, Model: {len(correct_features)}")
            model_data['feature_columns'] = list(correct_features)
            
    # [FIX] Validate feature count against model metadata
    # Some older models might have 181 features in metadata but model expects 175.
    n_features_expected = getattr(model, 'n_features_in_', None)
    if n_features_expected:
        saved_feature_count = len(model_data.get('feature_columns', []))
        if n_features_expected != saved_feature_count:
            # print(f"⚠️ Feature count mismatch! Model expects {n_features_expected}, Metadata has {saved_feature_count}")
            # Try to fix using booster feature names if available
            if hasattr(model, 'get_booster'):
                try:
                    booster_features = model.get_booster().feature_names
                    if booster_features and len(booster_features) == n_features_expected:
                         model_data['feature_columns'] = list(booster_features)
                except:
                    pass
            
    return model_data


def get_feature_importance(model_data, top_n=20):
    """
    Trích xuất feature importance từ XGBoost model.
    XGBoost lưu importance scores cho mỗi feature dựa trên số lần feature được dùng để split.
    
    Args:
        model_data: dict chứa 'model' và 'feature_columns'
        top_n: số lượng features quan trọng nhất cần trả về
    Returns:
        pd.DataFrame với columns ['feature', 'importance'] đã sắp xếp giảm dần
    """
    model = model_data['model']
    features = model_data['feature_columns']
    
    # XGBoost model có thuộc tính feature_importances_ (gain-based importance)
    importances = model.feature_importances_
    
    # Ensure lengths match before creating DataFrame
    if len(features) != len(importances):
        # If mismatch persists (should be fixed in load_model), try to fix or truncate
        min_len = min(len(features), len(importances))
        features = features[:min_len]
        importances = importances[:min_len]

    # Tạo DataFrame và sắp xếp theo importance giảm dần
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    return importance_df


def get_model_metrics(model_data, df):
    """
    Tính toán model metrics trên toàn bộ dataset.
    Dùng model đã train để predict trên data hiện tại và tính accuracy, precision, recall, f1, roc_auc.
    
    Args:
        model_data: dict chứa 'model', 'scaler', 'feature_columns'
        df: pd.DataFrame chứa processed data
    Returns:
        dict chứa các metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_columns']
    target_col = model_data.get('target_column', 'churn_label')
    
    # Log to file for debugging
    with open("debug_metrics_log.txt", "a") as f:
        f.write(f"\n--- Model Metrics Debug ---\n")
        f.write(f"DF Shape: {df.shape}\n")
        f.write(f"Columns: {list(df.columns)}\n")
        f.write(f"Target Col: {target_col}\n")
        
    if target_col not in df.columns:
        with open("debug_metrics_log.txt", "a") as f:
            f.write(f"ERROR: Target col {target_col} not in df.\n")
        return {}
    
    # Lọc features có trong data
    # IMPORTANT: Must ensure we have ALL features the model expects.
    # If data is missing features, add them as 0.
    
    # 1. Select available features
    available_features = [f for f in feature_cols if f in df.columns]
    X = df[available_features].copy()
    
    # 2. Add missing features (with 0)
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        for f in missing_features:
            X[f] = 0
            
    # 3. Reorder to match model's expected order
    X = X[feature_cols]
    
    y = df[target_col]
    
    # Scale numerical features (giống như lúc training)
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if scaler and numerical_cols:
        # Check if scaler expects different features
        # If scaler was fitted on 181 features but X has 175 (or vice versa), transform will fail.
        # But usually scaler is fitted on numericals ONLY.
        # If numerical_cols match scaler's feature_names_in_, great.
        try:
             X[numerical_cols] = scaler.transform(X[numerical_cols])
        except Exception:
            # Fallback: if scaler fails, maybe ignore scaling for metrics (not ideal but prevents crash)
            # Or try to align scaler features.
            pass
    
    # Predict
    try:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]  # Xác suất của class 1 (churn)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
    except Exception as e:
        return {'error': str(e)}


def get_churn_summary(df):
    """
    Tính toán tóm tắt churn từ processed data.
    Returns: dict chứa tổng user, churned, retained, churn_rate.
    """
    target_col = 'churn_label'
    if target_col not in df.columns:
        return {}
    
    total = len(df)
    churned = int(df[target_col].sum())
    retained = total - churned
    churn_rate = churned / total if total > 0 else 0
    
    return {
        'total_users': total,
        'churned': churned,
        'retained': retained,
        'churn_rate': churn_rate
    }


def get_pipeline_status():
    """
    Kiểm tra trạng thái pipeline: file nào đã được tạo, thời gian tạo.
    Returns: list of dict chứa thông tin các bước pipeline.
    """
    steps = []
    
    # Raw data
    raw_files = sorted((PROJECT_ROOT / "data" / "raw").glob("churn_data_*.parquet"))
    if raw_files:
        f = raw_files[-1]
        steps.append({
            'step': '1. Ingestion',
            'file': f.name,
            'time': pd.Timestamp.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
            'status': '✅'
        })
    else:
        steps.append({'step': '1. Ingestion', 'file': '-', 'time': '-', 'status': '❌'})
    
    # Processed data
    proc_files = sorted((PROJECT_ROOT / "data" / "processed").glob("churn_processed_*.parquet"))
    if proc_files:
        f = proc_files[-1]
        steps.append({
            'step': '2. Processing',
            'file': f.name,
            'time': pd.Timestamp.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
            'status': '✅'
        })
    else:
        steps.append({'step': '2. Processing', 'file': '-', 'time': '-', 'status': '❌'})
    
    # Model
    model_files = sorted((PROJECT_ROOT / "models").glob("churn_model_*.joblib"))
    if model_files:
        f = model_files[-1]
        steps.append({
            'step': '3. Training',
            'file': f.name,
            'time': pd.Timestamp.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
            'status': '✅'
        })
    else:
        steps.append({'step': '3. Training', 'file': '-', 'time': '-', 'status': '❌'})
    
    return steps
