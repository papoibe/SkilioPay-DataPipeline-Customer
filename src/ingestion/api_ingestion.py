"""
API Data Ingestion Module

Handles data ingestion from REST APIs with rate limiting and error handling.
Supports pagination and incremental data loading.
"""

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
import logging
import time
from datetime import datetime, timedelta
import json

from utils.config import ConfigManager
from utils.logging_config import get_logger
from utils.data_validation import DataValidator

logger = get_logger(__name__)


class APIIngestion:
    """API data ingestion handler"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize API ingestion
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validator = DataValidator()
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(
            rate_limit=config.get("data_sources", {}).get("api", {}).get("rate_limit", 100),
            time_window=60  # 1 minute
        )
        
        # Configure session
        self.session.timeout = config.get("data_sources", {}).get("api", {}).get("timeout", 30)
        
    def ingest_from_api(
        self,
        endpoint: str,
        output_path: str,
        params: Optional[Dict[str, Any]] = None,
        pagination: bool = True,
        incremental: bool = False,
        last_update: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Ingest data from API endpoint
        
        Args:
            endpoint: API endpoint path
            output_path: Path to output Parquet file
            params: Query parameters
            pagination: Whether to handle pagination
            incremental: Whether to do incremental load
            last_update: Last update timestamp for incremental load
            
        Returns:
            Dictionary with ingestion metadata
        """
        try:
            logger.info(f"Starting API ingestion from {endpoint}")
            
            # Prepare parameters
            if params is None:
                params = {}
            
            if incremental and last_update:
                params['updated_since'] = last_update.isoformat()
            
            # Fetch data
            all_data = []
            if pagination:
                for page_data in self._fetch_paginated_data(endpoint, params):
                    all_data.extend(page_data)
            else:
                data = self._fetch_single_page(endpoint, params)
                all_data = data
            
            logger.info(f"Successfully fetched {len(all_data)} records from API")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Add metadata
            df = self._add_metadata(df, endpoint)
            
            # Convert to Parquet
            parquet_path = self._convert_to_parquet(df, output_path)
            logger.info(f"Successfully saved to Parquet: {parquet_path}")
            
            # Generate metadata
            metadata = self._generate_metadata(df, endpoint, parquet_path)
            
            logger.info("API ingestion completed successfully")
            return metadata
            
        except Exception as e:
            logger.error(f"API ingestion failed: {str(e)}")
            raise
    
    def _fetch_paginated_data(
        self, 
        endpoint: str, 
        params: Dict[str, Any]
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Fetch data with pagination support"""
        base_url = self.config.get("data_sources", {}).get("api", {}).get("base_url", "")
        url = f"{base_url}{endpoint}"
        
        page = 1
        page_size = 100  # Default page size
        
        while True:
            # Add pagination parameters
            page_params = params.copy()
            page_params.update({
                'page': page,
                'page_size': page_size
            })
            
            # Rate limiting
            self.rate_limiter.wait()
            
            # Make request
            response = self.session.get(url, params=page_params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if we have data
            if not data or len(data) == 0:
                break
            
            yield data
            
            # Check if there are more pages
            if len(data) < page_size:
                break
            
            page += 1
    
    def _fetch_single_page(self, endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch single page of data"""
        base_url = self.config.get("data_sources", {}).get("api", {}).get("base_url", "")
        url = f"{base_url}{endpoint}"
        
        # Rate limiting
        self.rate_limiter.wait()
        
        # Make request
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def _add_metadata(self, df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
        """Add metadata columns to dataframe"""
        df = df.copy()
        df['_ingestion_timestamp'] = datetime.now().isoformat()
        df['_source_endpoint'] = endpoint
        df['_row_number'] = range(len(df))
        return df
    
    def _convert_to_parquet(self, df: pd.DataFrame, output_path: str) -> str:
        """Convert DataFrame to Parquet format"""
        # Create output directory if not exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)
        
        # Write to Parquet
        # removesuffix: chỉ xóa suffix chính xác, không xóa từng ký tự như rstrip
        parquet_path = f"{output_path.removesuffix('.parquet')}.parquet"
        pq.write_table(
            table, 
            parquet_path,
            compression='snappy'
        )
        
        return parquet_path
    
    def _generate_metadata(self, df: pd.DataFrame, endpoint: str, output_path: str) -> Dict[str, Any]:
        """Generate ingestion metadata"""
        return {
            "ingestion_timestamp": datetime.now().isoformat(),
            "source_endpoint": endpoint,
            "output_file": output_path,
            "row_count": len(df),
            "column_count": len(df.columns),
            "file_size_bytes": Path(output_path).stat().st_size if Path(output_path).exists() else 0,
            "columns": list(df.columns),
            "status": "success"
        }


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, rate_limit: int, time_window: int = 60):
        """
        Initialize rate limiter
        
        Args:
            rate_limit: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.requests = []
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        
        # Remove old requests outside time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        # If we're at the rate limit, wait
        if len(self.requests) >= self.rate_limit:
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                # Clean up old requests after sleep
                now = time.time()
                self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        # Record this request
        self.requests.append(now)


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Data Ingestion")
    parser.add_argument("--endpoint", required=True, help="API endpoint")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--params", help="JSON string of query parameters")
    parser.add_argument("--no-pagination", action="store_true", help="Disable pagination")
    parser.add_argument("--incremental", action="store_true", help="Enable incremental loading")
    parser.add_argument("--last-update", help="Last update timestamp for incremental loading")
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Parse parameters
    params = json.loads(args.params) if args.params else None
    
    # Parse last update
    last_update = None
    if args.last_update:
        last_update = datetime.fromisoformat(args.last_update)
    
    # Initialize ingestion
    ingestion = APIIngestion(config)
    
    # Run ingestion
    metadata = ingestion.ingest_from_api(
        endpoint=args.endpoint,
        output_path=args.output,
        params=params,
        pagination=not args.no_pagination,
        incremental=args.incremental,
        last_update=last_update
    )
    
    print(f"API ingestion completed: {metadata}")


if __name__ == "__main__":
    main()
