
# kiểm tra chất lượng dữ liệu 
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)
# Chạy nhiều loại check chất lượng dữ liệu:
# completeness (độ đầy đủ)
# uniqueness (tính duy nhất)
# validity (hợp lệ)
# consistency (nhất quán)
# outliers (điểm ngoại lai)
# distribution (phân phối dữ liệu)
# Tính overall_score và đánh dấu PASS/FAIL

class DataQualityChecker:
    def __init__(self):
        self.quality_metrics = {}
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% completeness required
            'uniqueness': 0.99,    # 99% uniqueness required
            'validity': 0.98,      # 98% validity required
            'consistency': 0.95    # 95% consistency required
        }
    
    # Hàm chính, nhận DataFrame và trả về report dạng dict
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting data quality checks")
        
        # tạo khung 
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'checks': {}
        }
        
        # Run all quality checks
        quality_report['checks']['completeness'] = self._check_completeness(df)
        quality_report['checks']['uniqueness'] = self._check_uniqueness(df)
        quality_report['checks']['validity'] = self._check_validity(df)
        quality_report['checks']['consistency'] = self._check_consistency(df)
        quality_report['checks']['outliers'] = self._check_outliers(df)
        quality_report['checks']['distribution'] = self._check_distribution(df)
        
        # tính overall score
        quality_report['overall_score'] = self._calculate_overall_score(quality_report['checks'])
        quality_report['status'] = 'PASS' if quality_report['overall_score'] >= 0.8 else 'FAIL'
        
        logger.info(f"Data quality check completed. Overall score: {quality_report['overall_score']:.3f}")
        
        return quality_report
    
    # tính độ đầy đủ
    # Tính tỉ lệ không bị null trên toàn dataset và từng cột.
    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        completeness_report = {
            'score': 0.0,
            'issues': [],
            'column_details': {}
        }
        
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_score = 1 - (missing_cells / total_cells)
        
        completeness_report['score'] = completeness_score
        
        # Check individual columns
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_rate = missing_count / len(df)
            
            completeness_report['column_details'][column] = {
                'missing_count': int(missing_count),
                'missing_rate': missing_rate,
                'completeness_rate': 1 - missing_rate
            }
            
            if missing_rate > 0.1:  # More than 10% missing
                completeness_report['issues'].append(
                    f"Column '{column}' has {missing_rate:.1%} missing values"
                )
        
        return completeness_report
    
    # tính độ duy nhất
    def _check_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        uniqueness_report = {
            'score': 0.0,
            'issues': [],
            'column_details': {}
        }
        
        # Check for duplicate rows
        # Tính uniqueness cho cột key (ở đây mặc định là user_id nếu tồn tại)
        duplicate_rows = df.duplicated().sum()
        duplicate_rate = duplicate_rows / len(df)
        
        uniqueness_report['duplicate_rows'] = int(duplicate_rows) # số lượng bản ghi trùng lặp
        uniqueness_report['duplicate_rate'] = duplicate_rate # tỉ lệ trùng lặp
        
        # Check key columns for uniqueness
        key_columns = ['user_id'] if 'user_id' in df.columns else [] # cột key (ở đây mặc định là user_id nếu tồn tại)
        
        for column in key_columns:
            unique_count = df[column].nunique()
            total_count = len(df)
            uniqueness_rate = unique_count / total_count
            
            uniqueness_report['column_details'][column] = {
                'unique_count': unique_count,
                'total_count': total_count,
                'uniqueness_rate': uniqueness_rate
            }
            
            if uniqueness_rate < 0.99:  # Less than 99% unique
                uniqueness_report['issues'].append(
                    f"Column '{column}' has {uniqueness_rate:.1%} uniqueness"
                )
        
        # Calculate overall uniqueness score
        if key_columns:
            avg_uniqueness = np.mean([
                uniqueness_report['column_details'][col]['uniqueness_rate'] 
                for col in key_columns
            ])
            uniqueness_report['score'] = avg_uniqueness * (1 - duplicate_rate)
        else:
            uniqueness_report['score'] = 1 - duplicate_rate
        
        return uniqueness_report
    
    # tính độ hợp lệ 
    def _check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        validity_report = {
            'score': 0.0,
            'issues': [],
            'column_details': {}
        }
        
        validity_issues = 0
        total_checks = 0
        
        # Check age validity
        if 'age' in df.columns:
            total_checks += 1
            invalid_ages = df[(df['age'] < 13) | (df['age'] > 100)]
            if len(invalid_ages) > 0:
                validity_issues += 1
                validity_report['issues'].append(f"Invalid age values: {len(invalid_ages)} records")
            validity_report['column_details']['age'] = {
                'invalid_count': len(invalid_ages),
                'valid_rate': 1 - (len(invalid_ages) / len(df))
            }
        
        # Check churn label validity
        if 'churn_label' in df.columns:
            total_checks += 1
            invalid_churn = df[~df['churn_label'].isin([0, 1])]
            if len(invalid_churn) > 0:
                validity_issues += 1
                validity_report['issues'].append(f"Invalid churn_label values: {len(invalid_churn)} records")
            validity_report['column_details']['churn_label'] = {
                'invalid_count': len(invalid_churn),
                'valid_rate': 1 - (len(invalid_churn) / len(df))
            }
        
        # Check email rates validity
        email_columns = ['emails_open_rate_90d', 'emails_click_rate_90d']
        for col in email_columns:
            if col in df.columns:
                total_checks += 1
                invalid_rates = df[(df[col] < 0) | (df[col] > 1)]
                if len(invalid_rates) > 0:
                    validity_issues += 1
                    validity_report['issues'].append(f"Invalid {col} values: {len(invalid_rates)} records")
                validity_report['column_details'][col] = {
                    'invalid_count': len(invalid_rates),
                    'valid_rate': 1 - (len(invalid_rates) / len(df))
                }
        
        # Check GMV validity
        if 'gmv_2024' in df.columns:
            total_checks += 1
            negative_gmv = df[df['gmv_2024'] < 0]
            if len(negative_gmv) > 0:
                validity_issues += 1
                validity_report['issues'].append(f"Negative GMV values: {len(negative_gmv)} records")
            validity_report['column_details']['gmv_2024'] = {
                'invalid_count': len(negative_gmv),
                'valid_rate': 1 - (len(negative_gmv) / len(df))
            }
        
        validity_report['score'] = 1 - (validity_issues / total_checks) if total_checks > 0 else 1.0
        
        return validity_report
    
    # tính độ nhất quán 
    # Các rule:
    # Sessions 30d vs 90d
    # sessions_30d không được > sessions_90d.
    # Orders 30d vs 90d
    # orders_30d không được > orders_90d.
    # GMV vs AOV * Orders
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        consistency_report = {
            'score': 0.0,
            'issues': [],
            'column_details': {}
        }
        
        consistency_issues = 0
        total_checks = 0
        
        # Check session consistency
        if 'sessions_30d' in df.columns and 'sessions_90d' in df.columns:
            total_checks += 1
            inconsistent_sessions = df[df['sessions_30d'] > df['sessions_90d']]
            if len(inconsistent_sessions) > 0:
                consistency_issues += 1
                consistency_report['issues'].append(
                    f"Inconsistent sessions (30d > 90d): {len(inconsistent_sessions)} records"
                )
            consistency_report['column_details']['sessions_consistency'] = {
                'inconsistent_count': len(inconsistent_sessions),
                'consistency_rate': 1 - (len(inconsistent_sessions) / len(df))
            }
        
        # Check orders consistency
        if 'orders_30d' in df.columns and 'orders_90d' in df.columns:
            total_checks += 1
            inconsistent_orders = df[df['orders_30d'] > df['orders_90d']]
            if len(inconsistent_orders) > 0:
                consistency_issues += 1
                consistency_report['issues'].append(
                    f"Inconsistent orders (30d > 90d): {len(inconsistent_orders)} records"
                )
            consistency_report['column_details']['orders_consistency'] = {
                'inconsistent_count': len(inconsistent_orders),
                'consistency_rate': 1 - (len(inconsistent_orders) / len(df))
            }
        
        # Check GMV vs AOV consistency
        if 'gmv_2024' in df.columns and 'aov_2024' in df.columns and 'orders_2024' in df.columns:
            total_checks += 1
            calculated_gmv = df['aov_2024'] * df['orders_2024']
            gmv_diff = abs(df['gmv_2024'] - calculated_gmv)
            inconsistent_gmv = df[gmv_diff > 0.01]  # Allow small floating point differences
            if len(inconsistent_gmv) > 0:
                consistency_issues += 1
                consistency_report['issues'].append(
                    f"Inconsistent GMV calculation: {len(inconsistent_gmv)} records"
                )
            consistency_report['column_details']['gmv_consistency'] = {
                'inconsistent_count': len(inconsistent_gmv),
                'consistency_rate': 1 - (len(inconsistent_gmv) / len(df))
            }
        
        consistency_report['score'] = 1 - (consistency_issues / total_checks) if total_checks > 0 else 1.0
        
        return consistency_report

    # kiểm tra outliers    
    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        outliers_report = {
            'score': 0.0,
            'issues': [],
            'column_details': {}
        }
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        outlier_counts = []
        
        for column in numerical_columns:
            if column in ['churn_label', 'rfm_recency', 'rfm_frequency']:  # Skip categorical numerical
                continue
            
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_count = len(outliers)
            outlier_rate = outlier_count / len(df)
            
            outliers_report['column_details'][column] = {
                'outlier_count': outlier_count,
                'outlier_rate': outlier_rate,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if outlier_rate > 0.05:  # More than 5% outliers
                outliers_report['issues'].append(
                    f"Column '{column}' has {outlier_rate:.1%} outliers"
                )
            
            outlier_counts.append(outlier_rate)
        
        # Calculate overall outlier score
        if outlier_counts:
            avg_outlier_rate = np.mean(outlier_counts)
            outliers_report['score'] = 1 - avg_outlier_rate
        else:
            outliers_report['score'] = 1.0
        
        return outliers_report
    
    # kiểm tra phân phối dữ liệu
    def _check_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        distribution_report = {
            'score': 0.0,
            'issues': [],
            'column_details': {}
        }
        
        # Check target variable distribution
        if 'churn_label' in df.columns:
            churn_distribution = df['churn_label'].value_counts(normalize=True)
            churn_rate = churn_distribution.get(1, 0)
            
            distribution_report['column_details']['churn_label'] = {
                'churn_rate': churn_rate,
                'distribution': churn_distribution.to_dict()
            }
            
            if churn_rate < 0.05 or churn_rate > 0.5:  # Unbalanced dataset
                distribution_report['issues'].append(
                    f"Unbalanced churn rate: {churn_rate:.1%}"
                )
        
        # Check country distribution
        if 'country' in df.columns:
            country_distribution = df['country'].value_counts(normalize=True)
            max_country_share = country_distribution.max()
            
            distribution_report['column_details']['country'] = {
                'max_share': max_country_share,
                'distribution': country_distribution.to_dict()
            }
            
            if max_country_share > 0.8:  # One country dominates
                distribution_report['issues'].append(
                    f"Unbalanced country distribution: {max_country_share:.1%} in one country"
                )
        
        # Calculate distribution score
        distribution_report['score'] = 0.8  # Default score, can be refined based on specific requirements
        
        return distribution_report
    
    # Gán trọng số cho từng loại check:
    def _calculate_overall_score(self, checks: Dict[str, Any]) -> float:
        weights = {
            'completeness': 0.25,
            'uniqueness': 0.20,
            'validity': 0.25,
            'consistency': 0.20,
            'outliers': 0.10
        }
        
        overall_score = 0.0
        for check_name, weight in weights.items():
            if check_name in checks:
                overall_score += checks[check_name]['score'] * weight
        
        return overall_score


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Quality Check")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--output", help="Path to output quality report")
    
    args = parser.parse_args()
    
    # Load data
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    
    # Initialize quality checker
    checker = DataQualityChecker()
    
    # Run quality checks
    report = checker.check_data_quality(df)
    
    # Print results
    print(f"Data Quality Report:")
    print(f"Overall Score: {report['overall_score']:.3f}")
    print(f"Status: {report['status']}")
    print(f"Total Rows: {report['total_rows']}")
    print(f"Total Columns: {report['total_columns']}")
    
    # Print issues
    for check_name, check_result in report['checks'].items():
        if check_result['issues']:
            print(f"\n{check_name.upper()} Issues:")
            for issue in check_result['issues']:
                print(f"  - {issue}")
    
    # Save report if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nQuality report saved to {args.output}")


if __name__ == "__main__":
    main()
