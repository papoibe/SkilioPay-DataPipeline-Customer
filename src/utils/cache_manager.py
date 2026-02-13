# Mô-đun quản lý cache cho pipeline
# Áp dụng nguyên tắc 3: Caching Strategy
# Giúp giảm thời gian xử lý bằng cách lưu tạm kết quả tính toán tốn kém
import hashlib
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import logging

from .logging_config import get_logger

logger = get_logger(__name__)


class PipelineCache:
    # Cache manager để lưu tạm kết quả tính toán, tránh tính lại
    # Sử dụng file-based cache với TTL (Time To Live)

    # khởi tạo config:
    # cache_dir: Thư mục lưu cache files
    # default_ttl: Thời gian sống mặc định của cache (giây), mặc định 24h
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'invalidations': 0
        }
    
    def _get_cache_key(self, data_hash: str, operation: str, params: Dict = None) -> str:
        # Tạo cache key duy nhất từ hash của data, operation và parameters
        key_parts = [operation, data_hash]
        if params:
            params_str = json.dumps(params, sort_keys=True)
            key_parts.append(params_str)
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_data_hash(self, data: Any) -> str:
        # Tạo hash từ data (DataFrame, dict, hoặc string)
        if hasattr(data, '__hash__'):
            try:
                return str(hash(data))
            except TypeError:
                pass
        
        # Nếu là DataFrame, hash các giá trị
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                # Hash bằng cách hash index và shape
                return hashlib.md5(
                    f"{data.shape}_{data.index.tolist()}".encode()
                ).hexdigest()
        except ImportError:
            pass
        
        # Fallback: hash string representation
        return hashlib.md5(str(data).encode()).hexdigest()
    
    def get(self, operation: str, data: Any = None, params: Dict = None) -> Optional[Any]:
        # Lấy giá trị từ cache nếu còn hợp lệ
        data_hash = self._get_data_hash(data) if data is not None else "no_data"
        cache_key = self._get_cache_key(data_hash, operation, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            self.stats['misses'] += 1
            logger.debug(f"Cache miss: {operation}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Kiểm tra TTL
            timestamp = cached_data.get('timestamp', 0)
            ttl = cached_data.get('ttl', self.default_ttl)
            age = datetime.now().timestamp() - timestamp
            
            if age > ttl:
                # Cache đã hết hạn
                cache_file.unlink()
                self.stats['misses'] += 1
                logger.debug(f"Cache expired: {operation}")
                return None
            
            # Cache hit
            self.stats['hits'] += 1
            logger.info(f"Cache hit: {operation} (age: {age:.0f}s)")
            return cached_data.get('value')
            
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            self.stats['misses'] += 1
            return None
    
    def set(self, operation: str, value: Any, data: Any = None, 
            params: Dict = None, ttl: Optional[int] = None):
        # Lưu giá trị vào cache với TTL
        data_hash = self._get_data_hash(data) if data is not None else "no_data"
        cache_key = self._get_cache_key(data_hash, operation, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Dùng `is None` thay vì `or` vì ttl=0 là falsy nhưng vẫn là giá trị hợp lệ
        ttl = self.default_ttl if ttl is None else ttl
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'timestamp': datetime.now().timestamp(),
                    'ttl': ttl,
                    'operation': operation
                }, f)
            
            self.stats['sets'] += 1
            logger.info(f"Cache set: {operation} (TTL: "
                        f"{ttl}s)")
            
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def invalidate(self, operation: str = None, pattern: str = None):
        # Xóa cache theo operation hoặc pattern
        if operation:
            # Xóa tất cả cache của operation
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    # Đọc file trước, đóng file handle, rồi mới xóa
                    # Trên Windows, không thể xóa file đang mở (file lock)
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    # unlink phải nằm NGOÀI `with` để file handle được giải phóng trước
                    if cached_data.get('operation') == operation:
                        cache_file.unlink()
                        self.stats['invalidations'] += 1
                except Exception:
                    pass
            logger.info(f"Invalidated cache for operation: {operation}")
        
        elif pattern:
            # Xóa cache theo pattern trong tên file
            for cache_file in self.cache_dir.glob(f"*{pattern}*.pkl"):
                cache_file.unlink()
                self.stats['invalidations'] += 1
            logger.info(f"Invalidated cache matching pattern: {pattern}")
    
    def clear(self):
        # Xóa toàn bộ cache
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        self.stats['invalidations'] += count
        logger.info(f"Cleared {count} cache files")
    
    def get_stats(self) -> Dict[str, Any]:
        # Lấy thống kê cache (hit rate, miss rate, ...)
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        
        # Tính dung lượng cache
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'invalidations': self.stats['invalidations'],
            'hit_rate': f"{hit_rate:.2f}%",
            'cache_size_mb': f"{cache_size / (1024 * 1024):.2f}",
            'cache_files': len(list(self.cache_dir.glob("*.pkl")))
        }


def cache_result(operation: str, ttl: int = 3600, params: Dict = None):
    # Decorator để cache kết quả của function
    def decorator(func):
        cache = PipelineCache()
        
        def wrapper(*args, **kwargs):
            # Tạo data hash từ arguments
            data = args[0] if args else None
            
            # Lấy từ cache
            cached_value = cache.get(operation, data, params)
            if cached_value is not None:
                return cached_value
            
            # Tính toán
            result = func(*args, **kwargs)
            
            # Lưu vào cache
            cache.set(operation, result, data, params, ttl)
            
            return result
        
        return wrapper
    return decorator


# using test
if __name__ == "__main__":
    # Test cache
    cache = PipelineCache(cache_dir="test_cache")
    
    # Test 1: Cache đơn giản
    cache.set("test_op", {"result": "data"}, ttl=60)
    result = cache.get("test_op")
    print(f"Test 1: {result}")
    
    # Test 2: Stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Test 3: Invalidate
    cache.invalidate("test_op")
    result = cache.get("test_op")
    print(f"Test 3 (after invalidate): {result}")

