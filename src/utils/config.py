# modun quản lý cấu hình
# load YAML -> cho tất cả module
# tất cả modun chính gọi để lấy config
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# configmanager cho pipeline
class ConfigManager:

    # Hàm khởi tạo: thiết lập đường dẫn file cấu hình mặc định và biến lưu cấu hình
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = None
    
    # Hàm tải cấu hình từ file YAML, có hỗ trợ thay biến môi trường dạng ${VAR}
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        if config_path:
            self.config_path = config_path
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_content = file.read()
                
                # Substitute environment variables
                config_content = self._substitute_env_vars(config_content)
                
                # Load YAML
                self.config = yaml.safe_load(config_content)
                
            logger.info(f"Configuration loaded from {self.config_path}")
            return self.config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    # Hàm thay thế biến môi trường trong nội dung YAML (ví dụ ${DB_HOST:localhost})
    def _substitute_env_vars(self, content: str) -> str:
        import re
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_env_var(match):
            var_name = match.group(1)
            # Nếu có default value thì sử dụng default value, nếu không thì sử dụng empty string
            default_value = match.group(2) if match.group(2) is not None else "" 
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replace_env_var, content)
    
    # Hàm truy xuất giá trị cấu hình theo key chấm (vd: "api.port"), có giá trị mặc định
    def get(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            self.load_config()
        
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    # Hàm lấy cấu hình database theo loại (postgresql/mongodb...) từ YAML
    def get_database_config(self, db_type: str = "postgresql") -> Dict[str, Any]:

        return self.get(f"storage.database.{db_type}", {})
    
    # Hàm lấy nhóm cấu hình ML (huấn luyện, features, tham số...)
    def get_ml_config(self) -> Dict[str, Any]:
        return self.get("ml", {})
    
    # Hàm lấy nhóm cấu hình API (host, port, workers...)
    def get_api_config(self) -> Dict[str, Any]:

        return self.get("api", {})
    
    # Hàm lấy nhóm cấu hình monitoring (logging, metrics, alerts...)
    def get_monitoring_config(self) -> Dict[str, Any]:
        return self.get("monitoring", {})
    
    # Hàm kiểm tra file cấu hình đã đủ các mục bắt buộc chưa
    def validate_config(self) -> bool:
        if self.config is None:
            logger.error("Configuration not loaded")
            return False
        
        required_sections = [
            "data_sources",
            "processing",
            "storage",
            "ml",
            "api",
            "monitoring"
        ]
        # Kiểm tra xem các section có trong config không
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        logger.info("Configuration validation passed")
        return True
    
    # Hàm lưu cấu hình ra file YAML (dùng khi muốn ghi đè/bổ sung cấu hình)
    def save_config(self, config: Dict[str, Any], output_path: str):

        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    # Hàm tạo file mẫu .env.template chứa các biến môi trường cần thiết
    def create_env_template(self, output_path: str = ".env.template"):
        env_vars = [
            "# Database Configuration",
            "DB_HOST=localhost",
            "DB_PORT=5432",
            "DB_NAME=skilio_pay",
            "DB_USER=postgres",
            "DB_PASSWORD=password",
            "",
            "# Cloud Storage",
            "AWS_ACCESS_KEY_ID=your_access_key",
            "AWS_SECRET_ACCESS_KEY=your_secret_key",
            "",
            "# Email Configuration",
            "EMAIL_USERNAME=your_email",
            "EMAIL_PASSWORD=your_password",
            "",
            "# API Configuration",
            "API_HOST=0.0.0.0",
            "API_PORT=8000",
            "",
            "# Environment",
            "ENVIRONMENT=development"
        ]
        
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(env_vars))
            
            logger.info(f"Environment template created: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating environment template: {e}")
            raise


# Hàm main dùng để test nhanh: load/validate config và tạo file .env mẫu nếu cần
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Manager")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--create-env", action="store_true", help="Create environment template")
    
    args = parser.parse_args()
    
    # Initialize 
    config_manager = ConfigManager(args.config)
    
    # Load configuration
    config = config_manager.load_config()
    
    if args.validate:
        if config_manager.validate_config():
            print("Configuration is valid")
        else:
            print("Configuration validation failed")
    
    if args.create_env:
        config_manager.create_env_template()
        print("Environment template created")
    
    # Print some configuration values
    print(f"Database host: {config_manager.get('storage.database.postgresql.host')}")
    print(f"API port: {config_manager.get('api.port')}")
    print(f"ML algorithm: {config_manager.get('ml.model.algorithm')}")


if __name__ == "__main__":
    main()
