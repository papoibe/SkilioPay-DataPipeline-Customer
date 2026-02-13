# Performance Optimization Guide - 8 Nguyên Tắc Tối Ưu Hiệu Năng

## Đánh Giá Hiện Trạng Project

### 1. Optimize Data Pipeline - ĐÃ ÁP DỤNG TỐT

**Hiện trạng:**
- Pipeline chia thành: `ingestion` → `processing` → `storage` → `ml`
- Airflow DAG orchestrates các bước độc lập
- Mỗi module có trách nhiệm riêng biệt

**Điểm mạnh:**
- Chia nhỏ pipeline thành ETL rõ ràng
- DAG cho phép chạy song song một số task (warehouse + training)
- Dễ debug và maintain

**Cần cải thiện:**
```python
#  HIỆN TẠI: Đọc toàn bộ file vào memory
df = pd.read_parquet(input_path)  # Nếu file > RAM sẽ crash

# NÊN DÙNG: Chunking cho file lớn
def _extract_data_chunked(self, input_path: str, chunk_size: int = 100000):
    for chunk in pd.read_parquet(input_path, chunksize=chunk_size):
        yield chunk
```

**Đề xuất:**
1. Thêm chunking cho file lớn (>1GB)
2. Thêm partitioning theo date cho dữ liệu lớn
3. Thêm parallel processing cho các bước độc lập

---

###  2. Data Format Selection - ĐÃ ÁP DỤNG TỐT

**Hiện trạng:**
-  CSV → Parquet (đúng cho analytical workload)
-  Dùng compression 'snappy' (tốt cho balance speed/size)
-  PyArrow Table (tối ưu cho Parquet)

**Điểm mạnh:**
- Chọn đúng format columnar (Parquet) cho analytical queries
- Compression giúp giảm storage

**Cần cải thiện:**
```python
#  HIỆN TẠI: Chỉ dùng snappy
compression='snappy'

#  NÊN CÓ: Tuỳ chọn compression theo use case
def _convert_to_parquet(self, df, output_path, compression='snappy'):
    """
    - snappy: Nhanh, tốt cho đọc/ghi thường xuyên
    - gzip: Nén tốt hơn, tốt cho archival
    - zstd: Balance tốt nhất (nên dùng mặc định)
    """
    pq.write_table(
        table, 
        parquet_path,
        compression=compression,  # Cho phép config
        compression_level=3,      # Thêm level cho gzip/zstd
        row_group_size=128 * 1024 * 1024,  # 128MB row groups (tối ưu query)
        use_dictionary=True,       # Dictionary encoding cho categorical
        use_deprecated_int96_timestamps=False
    )
```

**Đề xuất:**
1. Thêm row group size optimization (128MB cho analytical queries)
2. Dictionary encoding cho categorical columns
3. Column pruning khi đọc (chỉ đọc cột cần thiết)

---

### 3. Caching Strategy - CHƯA CÓ

**Hiện trạng:**
- KHÔNG có caching layer nào
- Mỗi lần chạy đều tính lại từ đầu
- Feature engineering chạy lại mỗi lần

**Vấn đề:**
```python
#  Mỗi lần chạy đều tính lại feature engineering
df_processed = self.feature_engineer.engineer_features(df_processed)

#  Model training luôn tính lại từ đầu
X, y, feature_columns = trainer.prepare_data(df)
```

**Cần implement:**
```python
#  CẦN THÊM: Cache layer
import functools
import hashlib
import pickle
from pathlib import Path

class PipelineCache:
    """Cache kết quả tính toán để tránh tính lại"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, data_hash: str, operation: str) -> str:
        """Tạo cache key từ hash của data và operation"""
        key = f"{operation}_{data_hash}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str):
        """Lấy từ cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, key: str, value: Any, ttl: int = 86400):
        """Lưu vào cache với TTL"""
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'value': value,
                'timestamp': datetime.now().timestamp(),
                'ttl': ttl
            }, f)

# Sử dụng trong ETL
def engineer_features_cached(self, df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering với cache"""
    # Tạo hash của dataframe
    data_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    cache_key = self.cache.get_cache_key(data_hash, "feature_engineering")
    
    # Kiểm tra cache
    cached_result = self.cache.get(cache_key)
    if cached_result:
        logger.info("Using cached feature engineering result")
        return cached_result['value']
    
    # Tính toán
    result = self.feature_engineer.engineer_features(df)
    
    # Lưu cache
    self.cache.set(cache_key, result)
    return result
```

**Đề xuất:**
1. Cache feature engineering results (TTL 24h)
2. Cache model predictions (TTL 1h)
3. Cache query results (TTL 5 phút)
4. Invalidate cache khi data thay đổi

---

###  4. Query Optimization - ĐÃ CÓ NHƯNG CẦN CẢI THIỆN

**Hiện trạng:**
- Đã có index trên các cột quan trọng (churn_label, country, timestamp)
- Có VACUUM và ANALYZE
- Chunking khi load data (chunksize=10000)

**Điểm mạnh:**
- Index đã được tạo trên các cột filter thường dùng
- Có optimize_table method

**Cần cải thiện:**
```python
# HIỆN TẠI: Query không tối ưu
query = f"SELECT * FROM {self.schema}.{table_name}"  # SELECT * không tối ưu

# NÊN DÙNG: Chỉ SELECT cột cần thiết
query = f"""
    SELECT 
        user_id,
        churn_label,
        country,
        rfm_segment
    FROM {self.schema}.{table_name}
    WHERE churn_label = 1
    AND country = 'Vietnam'
    LIMIT 1000
"""

# HIỆN TẠI: Load toàn bộ vào memory
df = pd.read_parquet(processing_metadata['output_file'])

# NÊN DÙNG: Chunking + Filtering sớm
def load_data_chunked(self, file_path: str, filters: Dict = None):
    """Load data với filtering sớm"""
    # Chỉ đọc cột cần thiết
    columns = ['user_id', 'churn_label', 'country']  # Config từ config.yaml
    df = pd.read_parquet(file_path, columns=columns)
    
    # Filter sớm
    if filters:
        for col, value in filters.items():
            df = df[df[col] == value]
    
    return df
```

**Cần thêm:**
1. CTE thay vì subquery phức tạp
2. Query plan analysis
3. Column pruning khi đọc Parquet
4. Partition pruning nếu có partitioning

```python
# QUERY TỐI ƯU VỚI CTE
def get_churn_users_by_country(self, country: str, limit: int = 1000):
    """Query tối ưu với CTE"""
    query = f"""
    WITH filtered_users AS (
        SELECT 
            user_id,
            churn_label,
            country,
            rfm_segment,
            gmv_2024
        FROM {self.schema}.users_processed
        WHERE country = :country
        AND churn_label = 1
    ),
    ranked_users AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (ORDER BY gmv_2024 DESC) as rn
        FROM filtered_users
    )
    SELECT * FROM ranked_users
    WHERE rn <= :limit
    """
    return self.query_data(query, {'country': country, 'limit': limit})
```

---

## Kế Hoạch Cải Thiện

### Phase 1: Caching (Ưu tiên cao)
- [ ] Implement PipelineCache class
- [ ] Cache feature engineering results
- [ ] Cache model predictions
- [ ] Cache query results

### Phase 2: Query Optimization
- [ ] Thay SELECT * bằng column cụ thể
- [ ] Thêm CTE cho queries phức tạp
- [ ] Implement query plan logging
- [ ] Column pruning khi đọc Parquet

### Phase 3: Pipeline Optimization
- [ ] Chunking cho file lớn
- [ ] Parallel processing cho các bước độc lập
- [ ] Partitioning theo date
- [ ] Incremental processing

### Phase 4: Format Optimization
- [ ] Configurable compression (zstd/gzip/snappy)
- [ ] Row group size optimization
- [ ] Dictionary encoding cho categorical
- [ ] Partition columns trong Parquet

---

## Metrics Cần Theo Dõi

1. **Cache Hit Rate**: % requests được phục vụ từ cache
2. **Query Execution Time**: Thời gian chạy query trung bình
3. **Memory Usage**: Peak memory khi xử lý
4. **Storage Size**: Dung lượng sau compression
5. **Pipeline Duration**: Tổng thời gian chạy pipeline

---

## Best Practices

### 1. Luôn filter sớm
```python
#  SAI
df = pd.read_parquet('data.parquet')
df = df[df['country'] == 'Vietnam']

# ĐÚNG
df = pd.read_parquet('data.parquet', filters=[('country', '==', 'Vietnam')])
```

### 2. Chỉ đọc cột cần thiết
```python
# SAI
df = pd.read_parquet('data.parquet')

#  ĐÚNG
df = pd.read_parquet('data.parquet', columns=['user_id', 'churn_label'])
```

### 3. Dùng index cho WHERE/JOIN
```python
# Đã có index
CREATE INDEX idx_users_churn ON users_processed(churn_label);
CREATE INDEX idx_users_country ON users_processed(country);

# Query sẽ nhanh hơn
SELECT * FROM users_processed WHERE churn_label = 1 AND country = 'Vietnam';
```

### 4. Cache kết quả tính toán tốn kém
```python
# Feature engineering tốn kém → cache lại
@cache_result(ttl=3600)  # Cache 1 giờ
def engineer_features(df):
    # Tính toán phức tạp
    return result
```

---

## Kết Luận

Project đã có nền tảng tốt với pipeline chia nhỏ và format đúng (Parquet). 
Cần ưu tiên implement caching và query optimization để tăng hiệu năng đáng kể.

**Ưu tiên:**
1. **Cao**: Caching layer (giảm 50-80% thời gian xử lý)
2. **Cao**: Query optimization (giảm 30-50% query time)
3. **Trung**: Chunking cho file lớn (tránh OOM)
4. **Thấp**: Format optimization (tối ưu thêm 10-20%)

