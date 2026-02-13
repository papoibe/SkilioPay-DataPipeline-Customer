# Model Training Module
# Xử lý huấn luyện, validation và đánh giá mô hình dự đoán churn
# Hỗ trợ nhiều thuật toán ML và cung cấp metrics đánh giá toàn diện
# Luồng: ETL Pipeline -> prepare data -> train model -> lưu model + metric vào MLflow -> mô hình sẵn sàng
# Thuật toán hỗ trợ: XGBoost, LightGBM, Random Forest

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
# from lightgbm import LGBMClassifier  # Tạm thời comment do lỗi với dask
from sklearn.ensemble import RandomForestClassifier
import optuna
import mlflow
import mlflow.sklearn

from utils.logging_config import get_logger
from utils.config import ConfigManager

logger = get_logger(__name__)


class ModelTrainer:
    # Model trainer cho churn prediction - xử lý toàn bộ quá trình huấn luyện mô hình
    
    def __init__(self, config: Dict[str, Any]):
        # Khởi tạo model trainer
        # Args:
        #   config: Dictionary chứa cấu hình (từ config.yaml)
        self.config = config
        self.ml_config = config.get("ml", {})
        self.model = None
        # StandardScaler: Thuật toán chuẩn hóa dữ liệu (Z-score normalization)
        # Công thức: z = (x - mean) / std
        # Mục đích: Đưa tất cả features về cùng scale (mean=0, std=1) để tránh features có giá trị lớn chi phối mô hình
        # Áp dụng cho: Các thuật toán nhạy cảm với scale như SVM, Neural Networks, và một số tree-based models
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = self.ml_config.get("features", {}).get("target", "churn_label")
        
        # Initialize MLflow - Hệ thống tracking experiments và model versioning
        # Lưu trữ: Parameters, metrics, artifacts (models) để so sánh và reproduce kết quả
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("churn_prediction")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        # Chuẩn bị dữ liệu cho huấn luyện
        # Lấy các cột feature (số và phân loại), thêm feature đã engineering, xử lý giá trị thiếu, encode categorical
        # Args:
        #   df: DataFrame đầu vào
        # Returns:
        #   Tuple (X: feature matrix, y: target vector, feature_columns: danh sách cột feature)
        logger.info("Preparing data for training")
        
        # Get feature columns
        categorical_features = self.ml_config.get("features", {}).get("categorical", [])
        numerical_features = self.ml_config.get("features", {}).get("numerical", [])
        
        # Combine all features
        all_features = categorical_features + numerical_features
        
        # Filter features that exist in dataframe
        available_features = [col for col in all_features if col in df.columns]
        
        # Add engineered features
        engineered_features = [col for col in df.columns if col not in all_features and col != self.target_column]
        available_features.extend(engineered_features)
        
        # Remove metadata columns và user_id (không phải feature)
        metadata_columns = [col for col in available_features if col.startswith('_')]
        available_features = [col for col in available_features if col not in metadata_columns and col != 'user_id']
        
        # Dedup features (remove duplicates while preserving order)
        available_features = list(dict.fromkeys(available_features))
        
        self.feature_columns = available_features
        
        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Xử lý missing values: Thuật toán imputation (điền giá trị thiếu)
        # Với numerical features: Dùng median (giá trị giữa) - robust với outliers hơn mean
        # Với categorical features: Dùng mode (giá trị xuất hiện nhiều nhất)
        # Lý do: Tránh mất mát thông tin khi xóa rows có missing values, giữ được kích thước dataset
        # Fillna riêng cho từng loại: numerical dùng median, categorical dùng mode
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        if len(numerical_cols) > 0:
            X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
        
        # Encode categorical variables
        X = self._encode_categorical_features(X)
        
        # Update feature columns after encoding (quan trọng!)
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Prepared {len(X)} samples with {len(self.feature_columns)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, self.feature_columns
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Mã hóa các feature phân loại bằng thuật toán One-Hot Encoding
        # Cách hoạt động: Mỗi category value được chuyển thành 1 binary column (0 hoặc 1)
        # Ví dụ: country ['VN', 'US', 'JP'] -> country_VN, country_US (drop_first=True để tránh multicollinearity)
        # Lý do: ML models không thể xử lý trực tiếp text/categorical data, cần chuyển sang số
        # drop_first=True: Bỏ 1 category để tránh dummy variable trap (perfect multicollinearity)
        X_encoded = X.copy()
        
        # Lấy tất cả categorical columns (object, category) trừ user_id
        categorical_cols = [col for col in X_encoded.columns 
                           if col != 'user_id' and 
                           (X_encoded[col].dtype == 'object' or 
                            X_encoded[col].dtype.name == 'category')]
        
        for col in categorical_cols:
            # One-hot encoding: Tạo binary columns cho mỗi category value
            # prefix=col: Thêm tên cột gốc vào tên cột mới (vd: country_VN)
            # drop_first=True: Bỏ category đầu tiên để tránh multicollinearity
            # dtype=int: Đảm bảo output là int (0/1) thay vì bool (pandas mới default là bool)
            dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True, dtype=int)
            # Nối các binary columns mới vào dataframe, xóa cột categorical gốc
            X_encoded = pd.concat([X_encoded.drop(col, axis=1), dummies], axis=1)
        
        return X_encoded
    
    def train_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        algorithm: str = None,
        hyperparameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # Huấn luyện mô hình với thuật toán được chỉ định
        # Chia dữ liệu thành train, validation, test, scale features, huấn luyện mô hình
        # Tính toán metrics, log metrics, log model, save model
        # Args:
        #   X: Feature matrix
        #   y: Target vector
        #   algorithm: Thuật toán ML sử dụng (mặc định từ config)
        #   hyperparameters: Hyperparameters cho thuật toán (mặc định từ config)
        # Returns:
        #   Dictionary chứa kết quả huấn luyện (metrics, feature importance, model path)
        logger.info("Starting model training")
        
        # Get algorithm from config if not specified
        if algorithm is None:
            algorithm = self.ml_config.get("model", {}).get("algorithm", "xgboost")
        
        # Get hyperparameters
        if hyperparameters is None:
            hyperparameters = self.ml_config.get("hyperparameters", {}).get(algorithm, {})
        
        # Chia dữ liệu: Thuật toán train_test_split với Stratified Sampling
        # Stratified sampling: Giữ nguyên tỷ lệ phân bố target variable trong mỗi set
        # Lý do: Đảm bảo train/val/test đều có cùng tỷ lệ churn (0/1), tránh bias
        # Tỷ lệ: 60% train, 20% validation, 20% test (chuẩn trong ML)
        test_size = self.ml_config.get("model", {}).get("test_size", 0.2)
        validation_size = self.ml_config.get("model", {}).get("validation_size", 0.2)
        random_state = self.ml_config.get("model", {}).get("random_state", 42)
        
        # Bước 1: Tách test set (20%) - dùng để đánh giá cuối cùng, không được chạm vào khi tuning
        # stratify=y: Đảm bảo tỷ lệ churn trong test set giống với toàn bộ dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Bước 2: Tách validation set (20% từ 80% còn lại) - dùng để tune hyperparameters
        # validation_size/(1-test_size): Tính lại tỷ lệ từ phần train còn lại
        # stratify=y_train: Giữ tỷ lệ churn trong validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size/(1-test_size), 
            random_state=random_state, stratify=y_train
        )
        
        # Chuẩn hóa features: Áp dụng StandardScaler
        # fit_transform(X_train): Tính mean và std từ train set, rồi transform train set
        # transform(X_val/X_test): Chỉ transform validation/test bằng mean và std đã tính từ train
        # Lý do: Tránh data leakage - không được dùng thông tin từ val/test để chuẩn hóa train
        # Kết quả: Tất cả features có mean ≈ 0, std ≈ 1, giúp mô hình học tốt hơn
        # Chỉ scale numerical columns, giữ nguyên categorical (đã được one-hot encoded)
        # Đảm bảo tất cả columns đều là numeric (categorical đã được one-hot encoded)
        # Chỉ lấy các columns numeric để tránh lỗi với XGBoost
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Scaling {len(numerical_cols)} numerical features out of {len(X_train.columns)} total features")
        
        # [FIX] Update feature_columns to match the actual text features used for training
        # This ensures the list saved in joblib matches model.n_features_in_
        self.feature_columns = numerical_cols

        
        X_train_scaled = X_train[numerical_cols].copy()
        X_val_scaled = X_val[numerical_cols].copy()
        X_test_scaled = X_test[numerical_cols].copy()
        
        # Scale tất cả columns (vì đã filter chỉ numeric)
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_scaled),
            columns=numerical_cols,
            index=X_train_scaled.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val_scaled),
            columns=numerical_cols,
            index=X_val_scaled.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_scaled),
            columns=numerical_cols,
            index=X_test_scaled.index
        )
        
        # Initialize model
        model = self._get_model(algorithm, hyperparameters)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(hyperparameters)
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("validation_samples", len(X_val))
            mlflow.log_param("test_samples", len(X_test))
            
            # Huấn luyện mô hình: Áp dụng thuật toán được chọn
            if algorithm == "xgboost":
                # XGBoost: Gradient Boosting Decision Trees với regularization
                # Thuật toán: Xây dựng nhiều decision trees tuần tự, mỗi tree sửa lỗi của tree trước
                # Ưu điểm: Xử lý tốt non-linear relationships, feature interactions, missing values
                # eval_set: Đánh giá trên validation set sau mỗi iteration
                # early_stopping_rounds: Dừng sớm nếu validation score không cải thiện trong 10 rounds
                # Lý do early stopping: Tránh overfitting, tiết kiệm thời gian training
                # XGBoost 3.x: early_stopping_rounds không còn được hỗ trợ trong fit()
                # Có thể set trong __init__ nếu cần, nhưng tạm thời bỏ để tránh lỗi
                # Convert sang numpy array để tránh lỗi với pandas DataFrame
                # XGBoost 3.x yêu cầu numpy array, không chấp nhận DataFrame
                X_train_array = np.asarray(X_train_scaled.values if isinstance(X_train_scaled, pd.DataFrame) else X_train_scaled, dtype=np.float32)
                X_val_array = np.asarray(X_val_scaled.values if isinstance(X_val_scaled, pd.DataFrame) else X_val_scaled, dtype=np.float32)
                y_train_array = np.asarray(y_train.values if isinstance(y_train, pd.Series) else y_train, dtype=np.int32)
                y_val_array = np.asarray(y_val.values if isinstance(y_val, pd.Series) else y_val, dtype=np.int32)
                
                model.fit(
                    X_train_array, y_train_array,
                    eval_set=[(X_val_array, y_val_array)],
                    verbose=False
                )
            else:
                # LightGBM hoặc Random Forest: Huấn luyện trực tiếp không có early stopping
                # LightGBM: Gradient boosting nhanh hơn XGBoost, dùng leaf-wise tree growth
                # Random Forest: Ensemble của nhiều decision trees độc lập, voting để quyết định
                # Convert sang numpy array để đảm bảo tương thích
                X_train_array = X_train_scaled.values if isinstance(X_train_scaled, pd.DataFrame) else X_train_scaled
                model.fit(X_train_array, y_train.values if isinstance(y_train, pd.Series) else y_train)
            
            # Dự đoán: Sử dụng mô hình đã huấn luyện để predict
            # predict(): Trả về class prediction (0 hoặc 1) - dùng threshold mặc định 0.5
            # predict_proba(): Trả về probability (0-1) - xác suất thuộc class 1 (churn)
            # [:, 1]: Lấy cột thứ 2 (probability của class 1 - churn)
            # Lý do cần cả 2: predict() cho class, predict_proba() cho metrics như ROC-AUC
            # Convert sang numpy array để tránh lỗi với XGBoost 3.x
            X_train_array = X_train_scaled.values if isinstance(X_train_scaled, pd.DataFrame) else X_train_scaled
            X_val_array = X_val_scaled.values if isinstance(X_val_scaled, pd.DataFrame) else X_val_scaled
            X_test_array = X_test_scaled.values if isinstance(X_test_scaled, pd.DataFrame) else X_test_scaled
            
            y_train_pred = model.predict(X_train_array)
            y_val_pred = model.predict(X_val_array)
            y_test_pred = model.predict(X_test_array)
            
            # Probability predictions: Cần cho ROC-AUC score và có thể điều chỉnh threshold
            y_train_proba = model.predict_proba(X_train_array)[:, 1]
            y_val_proba = model.predict_proba(X_val_array)[:, 1]
            y_test_proba = model.predict_proba(X_test_array)[:, 1]
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
            val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
            test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba)
            
            # Log metrics
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            for metric, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric}", value)
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Cross-validation
            cv_scores = self._cross_validate(model, X_train_scaled, y_train)
            mlflow.log_metrics({f"cv_{metric}": score for metric, score in cv_scores.items()})
            
            # Feature importance
            feature_importance = self._get_feature_importance(model)
            mlflow.log_params({f"feature_importance_{feat}": imp for feat, imp in feature_importance.items()})
            
            # Save model
            self.model = model
            
            # Prepare results
            results = {
                "algorithm": algorithm,
                "hyperparameters": hyperparameters,
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "cv_scores": cv_scores,
                "feature_importance": feature_importance,
                "model_path": f"models/churn_model_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            }
            
            # Save model to disk
            self._save_model(model, results["model_path"])
            
            logger.info("Model training completed successfully")
            return results
    
    def _get_model(self, algorithm: str, hyperparameters: Dict[str, Any]):
        # Tạo instance mô hình dựa trên thuật toán được chọn
        # random_state=42: Đảm bảo reproducibility - mỗi lần chạy cho kết quả giống nhau
        if algorithm == "xgboost":
            # XGBoost: Gradient Boosting với regularization (L1/L2)
            # Ưu điểm: Xử lý tốt missing values, feature interactions, non-linear patterns
            # Phù hợp: Dataset lớn, cần độ chính xác cao
            return xgb.XGBClassifier(**hyperparameters, random_state=42)
        elif algorithm == "lightgbm":
            # LightGBM: Gradient Boosting tối ưu tốc độ với leaf-wise tree growth
            # Ưu điểm: Nhanh hơn XGBoost, ít tốn memory, vẫn giữ độ chính xác cao
            # verbose=-1: Tắt log để output sạch hơn
            # Tạm thời comment do lỗi với dask
            raise ValueError("LightGBM is temporarily disabled due to dependency issues. Please use 'xgboost' or 'random_forest' instead.")
            # return LGBMClassifier(**hyperparameters, random_state=42, verbose=-1)
        elif algorithm == "random_forest":
            # Random Forest: Ensemble của nhiều decision trees độc lập
            # Cách hoạt động: Mỗi tree train trên subset data và features khác nhau (bootstrap + feature sampling)
            # Voting: Dự đoán cuối cùng = majority vote của tất cả trees
            # Ưu điểm: Robust với overfitting, không cần feature scaling
            return RandomForestClassifier(**hyperparameters, random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        # Tính toán các metrics đánh giá mô hình
        # Accuracy: Tỷ lệ dự đoán đúng = (TP + TN) / (TP + TN + FP + FN)
        # Precision: Độ chính xác khi dự đoán positive = TP / (TP + FP) - "Trong số dự đoán churn, bao nhiêu thực sự churn?"
        # Recall: Độ nhạy = TP / (TP + FN) - "Trong số thực sự churn, mô hình bắt được bao nhiêu?" (Quan trọng cho churn!)
        # F1: Harmonic mean của Precision và Recall = 2 * (Precision * Recall) / (Precision + Recall)
        # ROC-AUC: Diện tích dưới đường ROC curve - Đánh giá khả năng phân biệt giữa 2 classes (0-1), càng cao càng tốt
        # average='weighted': Tính trung bình có trọng số theo số lượng samples mỗi class
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1": f1_score(y_true, y_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_true, y_proba)
        }
    
    def _cross_validate(self, model, X: np.ndarray, y: pd.Series) -> Dict[str, float]:
        # Cross-Validation: Thuật toán đánh giá độ ổn định và generalization của mô hình
        # StratifiedKFold: Chia data thành 5 folds, mỗi fold giữ nguyên tỷ lệ target variable
        # Cách hoạt động: Train trên 4 folds, test trên 1 fold, lặp lại 5 lần
        # shuffle=True: Xáo trộn data trước khi chia để tránh bias theo thứ tự
        # Lý do: Đánh giá mô hình trên nhiều subsets khác nhau, phát hiện overfitting
        # Kết quả: Mean và std của 5 scores - mean cao = tốt, std thấp = ổn định
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # cross_val_score: Tự động train và evaluate trên mỗi fold, trả về list 5 scores
        # scoring='roc_auc': Metric dùng để đánh giá
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        return {
            "roc_auc_mean": cv_scores.mean(),  # Điểm trung bình - càng cao càng tốt
            "roc_auc_std": cv_scores.std()     # Độ lệch chuẩn - càng thấp càng ổn định
        }
    
    def _get_feature_importance(self, model) -> Dict[str, float]:
        # Feature Importance: Đo lường mức độ quan trọng của mỗi feature trong việc dự đoán
        # Tree-based models (XGBoost, LightGBM, RF): feature_importances_ dựa trên số lần feature được dùng và độ giảm impurity
        # Linear models: coef_ là trọng số, dùng absolute value để đo importance
        # Mục đích: Hiểu mô hình, feature selection, giải thích kết quả cho business
        if hasattr(model, 'feature_importances_'):
            # Tree-based models: Importance = tổng độ giảm impurity khi split trên feature này
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models: Dùng absolute coefficient làm importance
            importance = np.abs(model.coef_[0])
        else:
            return {}
        
        # Sắp xếp theo importance giảm dần và lấy top 20 features quan trọng nhất
        # Lý do top 20: Tập trung vào features có ảnh hưởng lớn nhất, dễ interpret
        feature_importance = dict(zip(self.feature_columns, importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])
    
    def _save_model(self, model, model_path: str):
        # Lưu mô hình vào disk sử dụng joblib (tối ưu cho scikit-learn models)
        # Lưu kèm: model, scaler, feature_columns, target_column
        # Lý do: Khi load lại để predict, cần scaler để transform data và biết features nào cần
        # joblib: Nhanh hơn pickle cho numpy arrays, tương thích tốt với scikit-learn
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu dictionary chứa tất cả components cần thiết để reproduce predictions
        joblib.dump({
            'model': model,  # Mô hình đã huấn luyện
            'scaler': self.scaler,  # Scaler đã fit - cần để transform data mới
            'feature_columns': self.feature_columns,  # Danh sách features - cần để đảm bảo đúng thứ tự
            'target_column': self.target_column  # Tên cột target - để reference
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        algorithm: str = "xgboost",
        n_trials: int = 100
    ) -> Dict[str, Any]:
        # Tối ưu hyperparameters sử dụng Optuna
        # Thử nhiều tổ hợp hyperparameters và chọn bộ tốt nhất
        # Args:
        #   X: Feature matrix
        #   y: Target vector
        #   algorithm: Thuật toán ML cần tối ưu
        #   n_trials: Số lần thử nghiệm (trials)
        # Returns:
        #   Dictionary chứa best_params, best_score, n_trials
        logger.info(f"Starting hyperparameter optimization for {algorithm}")
        
        def objective(trial):
            # Objective function: Hàm Optuna sẽ tối ưu để maximize
            # Optuna sử dụng Tree-structured Parzen Estimator (TPE) - Bayesian optimization
            # Cách hoạt động: Học từ các trials trước đó để suggest hyperparameters tốt hơn
            # Hiệu quả hơn Grid Search và Random Search vì tập trung vào vùng có tiềm năng
            
            # Định nghĩa không gian hyperparameters cần tối ưu
            if algorithm == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Số cây trong ensemble
                    'max_depth': trial.suggest_int('max_depth', 3, 10),  # Độ sâu tối đa của cây (sâu hơn = phức tạp hơn, dễ overfit)
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),  # Tốc độ học (thấp = chậm nhưng ổn định)
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Tỷ lệ samples dùng cho mỗi tree (0.6-1.0 = bagging)
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Tỷ lệ features dùng cho mỗi tree
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  # L1 regularization (Lasso) - giảm overfitting
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)  # L2 regularization (Ridge) - giảm overfitting
                }
            elif algorithm == "lightgbm":
                # LightGBM có hyperparameters tương tự XGBoost
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
            else:
                raise ValueError(f"Hyperparameter optimization not supported for {algorithm}")
            
            # Train và đánh giá mô hình với hyperparameters này
            # Dùng cross-validation để đánh giá khách quan, tránh overfitting trên validation set
            model = self._get_model(algorithm, params)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            
            # Trả về mean score - Optuna sẽ maximize giá trị này
            return cv_scores.mean()
        
        # Chạy optimization: Optuna sẽ thử n_trials lần và chọn best hyperparameters
        # direction='maximize': Tối ưu để maximize ROC-AUC score
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Hyperparameter optimization completed. Best score: {study.best_value}")
        
        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": n_trials
        }


def main():
    # Hàm main để chạy model training từ command line (CLI usage)
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--data", required=True, help="Path to training data (Parquet)")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--algorithm", help="ML algorithm to use")
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Load data
    df = pd.read_parquet(args.data)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Prepare data
    X, y, feature_columns = trainer.prepare_data(df)
    
    if args.optimize:
        # Optimize hyperparameters
        best_params = trainer.optimize_hyperparameters(X, y, args.algorithm, args.trials)
        print(f"Best hyperparameters: {best_params}")
        
        # Train with best parameters
        results = trainer.train_model(X, y, args.algorithm, best_params["best_params"])
    else:
        # Train model
        results = trainer.train_model(X, y, args.algorithm)
    
    print(f"Training completed: {results}")


if __name__ == "__main__":
    main()
