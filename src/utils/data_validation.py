# Mô-đun kiểm tra dữ liệu
# Cung cấp tiện ích xác thực dữ liệu dựa trên JSON schema
# và các luật nghiệp vụ tuỳ chỉnh cho dữ liệu dự đoán churn.

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from cerberus import Validator
import great_expectations as ge

logger = logging.getLogger(__name__)


class DataValidator:
    # Bộ xác thực dữ liệu cho pipeline dự đoán churn
    
    def __init__(self):
        # Khởi tạo validator (cerberus) và danh sách lỗi
        self.validator = Validator()
        self.validation_errors = []
    
    def validate_dataframe(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        # Xác thực DataFrame theo JSON schema (kiểm type, required, enum, pattern, range)
        try:
            # Convert DataFrame to records for validation
            records = df.to_dict('records')
            
            # Validate each record
            valid_records = []
            invalid_records = []
            
            for i, record in enumerate(records):
                if self.validator.validate(record, schema):
                    valid_records.append(record)
                else:
                    invalid_records.append({
                        'row': i, # row index
                        'data': record, # record data
                        'errors': self.validator.errors # validation errors
                    })
            
            # Calculate validation metrics
            total_records = len(records) 
            valid_count = len(valid_records)
            invalid_count = len(invalid_records)
            validation_rate = valid_count / total_records if total_records > 0 else 0
            
            # Create validation report
            result = {
                'is_valid': invalid_count == 0, # là dữ liệu hợp lệ
                'total_records': total_records, # tổng số dữ liệu
                'valid_records': valid_count, # số dữ liệu hợp lệ
                'invalid_records': invalid_count, # số dữ liệu không hợp lệ
                'validation_rate': validation_rate, # tỷ lệ dữ liệu hợp lệ
                'errors': [record['errors'] for record in invalid_records], # lỗi validation
                'invalid_data': invalid_records # dữ liệu không hợp lệ
            }
            
            if invalid_count > 0:
                logger.warning(f"Data validation found {invalid_count} invalid records out of {total_records}") # cảnh báo khi tìm thấy dữ liệu không hợp lệ
            else:
                logger.info(f"Data validation passed: {valid_count} records validated") # thông báo khi dữ liệu hợp lệ
            
            return result
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}") # lỗi khi validation failed
            return {
                'is_valid': False,
                'error': str(e), # lỗi
                'total_records': 0, # tổng số dữ liệu
                'valid_records': 0, # số dữ liệu hợp lệ
                'invalid_records': 0, # số dữ liệu không hợp lệ
                'validation_rate': 0.0, # tỷ lệ dữ liệu hợp lệ
                'errors': [str(e)] # lỗi validation
            }
    
    def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, Any]:
        # Kiểm tra kiểu dữ liệu các cột theo kỳ vọng (integer/number/string/...)
        type_errors = []
        
        for column, expected_type in expected_types.items():
            if column not in df.columns:
                type_errors.append(f"Column '{column}' not found") # cột không tìm thấy
                continue
            
            actual_type = str(df[column].dtype)
            
            # Map pandas types to expected types
            type_mapping = {
                'int64': 'integer', 
                'float64': 'number',
                'object': 'string',
                'bool': 'boolean',
                'datetime64[ns]': 'datetime'
            }
            
            mapped_actual = type_mapping.get(actual_type, actual_type)
            
            if mapped_actual != expected_type:
                type_errors.append(
                    f"Column '{column}': expected {expected_type}, got {mapped_actual}"
                )
        
        return {
            'is_valid': len(type_errors) == 0, # là dữ liệu hợp lệ
            'errors': type_errors, # lỗi validation
            'error_count': len(type_errors) # số lỗi validation
        }
    
    def validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Kiểm tra các luật nghiệp vụ cho dữ liệu churn (duy nhất, khoảng giá trị, nhãn, tỷ lệ...)
        business_errors = []
        
        # Rule 1: User ID should be unique
        if 'user_id' in df.columns:
            if df['user_id'].duplicated().any():
                business_errors.append("Duplicate user_id found") # lỗi khi tìm thấy user_id trùng lặp
        
        # Rule 2: Age should be between 13 and 100
        if 'age' in df.columns:
            invalid_ages = df[(df['age'] < 13) | (df['age'] > 100)] 
            if len(invalid_ages) > 0:
                business_errors.append(f"Invalid age values found: {len(invalid_ages)} records")
        
        # Rule 3: Churn label should be 0 or 1
        if 'churn_label' in df.columns:
            invalid_churn = df[~df['churn_label'].isin([0, 1])] # lỗi khi tìm thấy churn_label không phải là 0 hoặc 1
            if len(invalid_churn) > 0:
                business_errors.append(f"Invalid churn_label values found: {len(invalid_churn)} records")
        
        # Rule 4: GMV should be non-negative
        if 'gmv_2024' in df.columns:
            negative_gmv = df[df['gmv_2024'] < 0] # lỗi khi tìm thấy gmv_2024 âm
            if len(negative_gmv) > 0:
                business_errors.append(f"Negative GMV values found: {len(negative_gmv)} records")
        
        # Rule 5: Email rates should be between 0 and 1
        email_columns = ['emails_open_rate_90d', 'emails_click_rate_90d']
        for col in email_columns:
            if col in df.columns:
                invalid_rates = df[(df[col] < 0) | (df[col] > 1)]
                if len(invalid_rates) > 0:
                    business_errors.append(f"Invalid {col} values found: {len(invalid_rates)} records")
        
        return {
            'is_valid': len(business_errors) == 0,
            'errors': business_errors,
            'error_count': len(business_errors)
        }
    
    def validate_data_completeness(self, df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
        # Kiểm tra tính đầy đủ dữ liệu: đủ cột bắt buộc và không thiếu giá trị trong các cột đó
        completeness_errors = []
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            completeness_errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for missing values in required columns
        for col in required_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    completeness_errors.append(f"Column '{col}' has {missing_count} missing values")
                    
        # Create completeness report
        return {
            'is_valid': len(completeness_errors) == 0,
            'errors': completeness_errors,
            'error_count': len(completeness_errors)
        }


def main():
    # Hàm main để test nhanh: đọc dữ liệu, (tuỳ chọn) schema, chạy các kiểm tra
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Validation")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--schema", help="Path to JSON schema file")
    
    args = parser.parse_args()
    
    # Load data
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    
    # Initialize validator
    validator = DataValidator()
    
    # Load schema if provided
    schema = None
    if args.schema:
        with open(args.schema, 'r') as f:
            schema = json.load(f)
    
    # Run validations
    if schema:
        result = validator.validate_dataframe(df, schema)
        print(f"Schema validation: {result}")
    
    # Business rules validation
    business_result = validator.validate_business_rules(df)
    print(f"Business rules validation: {business_result}")
    
    # Data completeness validation
    required_columns = ['user_id', 'churn_label', 'age', 'country']
    completeness_result = validator.validate_data_completeness(df, required_columns)
    print(f"Completeness validation: {completeness_result}")


if __name__ == "__main__":
    main()
