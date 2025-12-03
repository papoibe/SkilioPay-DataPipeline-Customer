import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config # config dictionary
        self.label_encoders = {} # lưu encoder cho các cột ordinal (categorical có thứ tự)
        self.scalers = {} # lưu scaler cho các cột numerical
        
    #  xử lý dữ liệu thô thành dữ liệu đã xử lý
    #  Gọi tuần tự các block:
    # RFM
    # Hành vi (behavior)
    # Thời gian (temporal)
    # Tương tác (interaction)
    # Domain-specific (e-commerce)
    # Cuối cùng: encode categorical
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
       
        df_engineered = df.copy()
        logger.info("Starting feature engineering")
        # 1. RFM Analysis Features
        df_engineered = self._create_rfm_features(df_engineered)
        
        # 2. Behavioral Features
        df_engineered = self._create_behavioral_features(df_engineered)
        
        # 3. Temporal Features
        df_engineered = self._create_temporal_features(df_engineered)
        
        # 4. Interaction Features
        df_engineered = self._create_interaction_features(df_engineered)
        
        # 5. Domain-specific Features
        df_engineered = self._create_domain_features(df_engineered)
        
        # 6. Categorical Encoding
        df_engineered = self._encode_categorical_features(df_engineered)
        
        logger.info(f"Feature engineering completed. Added {len(df_engineered.columns) - len(df.columns)} new features")
        
        return df_engineered
    
    def _create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_rfm = df.copy()
        
        # RFM Segmentation
        df_rfm['rfm_segment'] = self._calculate_rfm_segment(
            df_rfm['rfm_recency'], 
            df_rfm['rfm_frequency'], 
            df_rfm['rfm_monetary']
        )
        
        # RFM Score (0-100 scale)
        df_rfm['rfm_score'] = (
            df_rfm['rfm_recency'] * 0.4 + 
            df_rfm['rfm_frequency'] * 0.3 + 
            df_rfm['rfm_monetary'] * 0.3
        )
        
        # RFM Categories
        df_rfm['rfm_category'] = pd.cut(
            df_rfm['rfm_score'], 
            bins=[0, 25, 50, 75, 100], 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        logger.info("RFM features created")
        return df_rfm
    
    def _calculate_rfm_segment(self, recency: pd.Series, frequency: pd.Series, monetary: pd.Series) -> pd.Series:
        r_quintile = self._quantile_bucket(recency, q=5, labels=[5, 4, 3, 2, 1], ascending=False)
        f_quintile = self._quantile_bucket(frequency, q=5, labels=[1, 2, 3, 4, 5], ascending=True)
        m_quintile = self._quantile_bucket(monetary, q=5, labels=[1, 2, 3, 4, 5], ascending=True)
        
        rfm_segment = r_quintile.astype(str) + f_quintile.astype(str) + m_quintile.astype(str)
        
        return rfm_segment

    def _quantile_bucket(self, series: pd.Series, q: int, labels: List[int], ascending: bool) -> pd.Series:
        unique_values = series.nunique()
        effective_q = min(q, unique_values)
        if effective_q < 2:
            fill_value = labels[0] if ascending else labels[-1]
            return pd.Series([fill_value] * len(series), index=series.index)
        
        ranks = series.rank(method="first", ascending=ascending)
        selected_labels = labels[:effective_q]
        return pd.qcut(ranks, effective_q, labels=selected_labels)
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_behavioral = df.copy()
        
        # Session intensity
        df_behavioral['session_intensity_30d'] = df_behavioral['sessions_30d'] / 30
        df_behavioral['session_intensity_90d'] = df_behavioral['sessions_90d'] / 90
        
        # Engagement ratio
        df_behavioral['engagement_ratio'] = (
            df_behavioral['sessions_30d'] / df_behavioral['sessions_90d'].replace(0, 1)
        )
        
        # Search activity
        df_behavioral['search_activity_ratio'] = (
            df_behavioral['search_queries_30d'] / df_behavioral['sessions_30d'].replace(0, 1)
        )
        
        # Page views per session
        df_behavioral['pages_per_session_30d'] = (
            df_behavioral['median_pages_viewed_30d'] * df_behavioral['sessions_30d']
        )
        
        # Email engagement
        df_behavioral['email_engagement_score'] = (
            df_behavioral['emails_open_rate_90d'] * 0.6 + 
            df_behavioral['emails_click_rate_90d'] * 0.4
        )
        
        # Support ticket ratio
        df_behavioral['support_intensity'] = (
            df_behavioral['support_tickets_2024'] / df_behavioral['orders_2024'].replace(0, 1)
        )
        
        logger.info("Behavioral features created")
        return df_behavioral
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_temporal = df.copy()
        
        # Registration recency
        df_temporal['reg_recency_category'] = pd.cut(
            df_temporal['reg_days'],
            bins=[0, 30, 90, 365, 1000, float('inf')],
            labels=['New', 'Recent', 'Established', 'Long-term', 'Veteran']
        )
        
        # Last order recency
        df_temporal['last_order_category'] = pd.cut(
            df_temporal['days_since_last_order'],
            bins=[0, 7, 30, 90, 180, float('inf')],
            labels=['Very Recent', 'Recent', 'Moderate', 'Old', 'Very Old']
        )
        
        # Order frequency
        df_temporal['order_frequency_2024'] = df_temporal['orders_2024'] / 365
        
        # Seasonal patterns (if we had date columns)
        # For now, create time-based features from reg_days
        df_temporal['is_weekend_reg'] = (df_temporal['reg_days'] % 7).isin([5, 6])
        df_temporal['is_month_end'] = (df_temporal['reg_days'] % 30) >= 25
        
        logger.info("Temporal features created")
        return df_temporal
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_interaction = df.copy()
        
        # Value per session
        df_interaction['value_per_session'] = (
            df_interaction['gmv_2024'] / df_interaction['sessions_90d'].replace(0, 1)
        )
        
        # Efficiency score (orders per session)
        df_interaction['order_efficiency'] = (
            df_interaction['orders_90d'] / df_interaction['sessions_90d'].replace(0, 1)
        )
        
        # Discount sensitivity
        df_interaction['discount_sensitivity'] = (
            df_interaction['discount_rate_2024'] * df_interaction['orders_2024']
        )
        
        # Quality score (CSAT * Review stars)
        df_interaction['quality_score'] = (
            df_interaction['avg_csat_2024'] * df_interaction['avg_review_stars_2024']
        )
        
        # Risk score (refund rate * support tickets)
        df_interaction['risk_score'] = (
            df_interaction['refund_rate_2024'] * df_interaction['support_tickets_2024']
        )
        
        # Engagement value (sessions * AOV)
        df_interaction['engagement_value'] = (
            df_interaction['sessions_90d'] * df_interaction['aov_2024']
        )
        
        logger.info("Interaction features created")
        return df_interaction
    
    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_domain = df.copy()
        
        # Customer lifetime value proxy
        df_domain['clv_proxy'] = (
            df_domain['gmv_2024'] * (365 / df_domain['reg_days'].replace(0, 1))
        )
        
        # Purchase consistency
        df_domain['purchase_consistency'] = (
            df_domain['orders_2024'] / (df_domain['reg_days'] / 30).replace(0, 1)
        )
        
        # Category diversity score
        df_domain['diversity_score'] = (
            df_domain['category_diversity_2024'] / df_domain['orders_2024'].replace(0, 1)
        )
        
        # App version adoption
        df_domain['is_latest_version'] = df_domain['app_version_major'].str.contains('3.x')
        
        # Device preference
        df_domain['is_mobile_heavy'] = df_domain['device_mix_ratio'] > 0.7
        
        # High-value customer flag
        df_domain['is_high_value'] = (
            (df_domain['gmv_2024'] > df_domain['gmv_2024'].quantile(0.8)) |
            (df_domain['aov_2024'] > df_domain['aov_2024'].quantile(0.8))
        )
        
        # At-risk customer flag
        df_domain['is_at_risk'] = (
            (df_domain['days_since_last_order'] > 90) |
            (df_domain['sessions_30d'] == 0) |
            (df_domain['refund_rate_2024'] > 0.1)
        )
        
        logger.info("Domain-specific features created")
        return df_domain
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        
        # Get categorical columns from config
        categorical_cols = self.config.get("ml", {}).get("features", {}).get("categorical", [])
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Label encoding for ordinal categories
                if col in ['reg_recency_category', 'last_order_category', 'rfm_category']:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_encoded[f"{col}_encoded"] = self.label_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    # One-hot encoding for nominal categories
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        logger.info("Categorical features encoded")
        return df_encoded
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")
        
        return dict(zip(feature_names, importance))
    
    def get_feature_correlation(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numerical_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        
        return correlations.dropna()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument("--input", required=True, help="Input Parquet file path")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    from ..utils.config import ConfigManager
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Load data
    df = pd.read_parquet(args.input)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(config)
    
    # Engineer features
    df_engineered = engineer.engineer_features(df)
    
    # Save results
    df_engineered.to_parquet(args.output, compression='snappy')
    
    print(f"Feature engineering completed. Output saved to {args.output}")
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_engineered.columns)}")


if __name__ == "__main__":
    main()
