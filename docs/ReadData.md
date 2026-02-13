# Hướng Dẫn Đọc Dữ Liệu Normalized trong PostgreSQL

## 1. Giải Thích Normalization (Standard Scaler)

### Công Thức
```
z = (x - mean) / std
```

### Kết Quả
- **Mean = 0**: Giá trị trung bình sau khi normalize = 0
- **Standard Deviation = 1**: Độ lệch chuẩn = 1

### Tại Sao Cần Normalize?
- Machine Learning models hoạt động tốt hơn với dữ liệu đã được chuẩn hóa
- Tránh các features có giá trị lớn (như GMV) chi phối các features có giá trị nhỏ (như age)
- Giúp model hội tụ nhanh hơn trong quá trình training

---

## 2. Cách Đọc Giá Trị Normalized

### 2.1. Giá Trị Âm (< 0)
**Ý nghĩa**: Thấp hơn trung bình (below average)

**Ví dụ**:
- `age = -1.23` → Tuổi nhỏ hơn trung bình khoảng 1.23 standard deviations
- `sessions_30d = -1.07` → Ít sessions hơn trung bình
- `gmv_2024 = -0.62` → GMV thấp hơn trung bình

### 2.2. Giá Trị Dương (> 0)
**Ý nghĩa**: Cao hơn trung bình (above average)

**Ví dụ**:
- `age = 1.63` → Tuổi lớn hơn trung bình khoảng 1.63 standard deviations
- `sessions_30d = 0.28` → Nhiều sessions hơn trung bình một chút
- `gmv_2024 = 2.43` → GMV rất cao (high value customer)

### 2.3. Giá Trị Gần 0 (-0.5 đến 0.5)
**Ý nghĩa**: Gần trung bình (near average)

**Ví dụ**:
- `age = 0.40` → Tuổi gần trung bình
- `sessions_30d = 0.28` → Số lượng sessions gần trung bình, hơi cao một chút

### 2.4. Giá Trị Tuyệt Đối Lớn (> 2 hoặc < -2)
**Ý nghĩa**: Outlier (rất cao hoặc rất thấp)

**Ví dụ**:
- `gmv_2024 = 2.43` → GMV rất cao (top users)
- `sessions_30d = -2.5` → Rất ít sessions (inactive users)

### 2.5. Categorical Columns
**Không bị normalize**, giữ nguyên giá trị gốc:
- `country`: "Thailand", "Indonesia", "Vietnam", etc.
- `city`: "Bangkok", "Jakarta", "Ho Chi Minh City", etc.
- `marketing_source`: "ads_fb", "organic", "referral", etc.

---

## 3. Ví Dụ Cụ Thể

### Ví Dụ 1: User U00001
```
user_id: U00001
age: -1.23 (normalized)
  → Tuổi nhỏ hơn trung bình (younger than average)
  
sessions_30d: 0.28 (normalized)
  → Số lượng sessions gần trung bình, hơi cao một chút
  
gmv_2024: -0.25 (normalized)
  → GMV thấp hơn trung bình một chút
  
country: "Thailand" (categorical - không normalize)
marketing_source: "ads_fb" (categorical - không normalize)
```

### Ví Dụ 2: User U00002
```
user_id: U00002
age: 1.63 (normalized)
  → Tuổi lớn hơn trung bình (older than average)
  
sessions_30d: 0.28 (normalized)
  → Số lượng sessions gần trung bình, hơi cao một chút
  
gmv_2024: -0.62 (normalized)
  → GMV thấp hơn trung bình
  
country: "Indonesia" (categorical - không normalize)
marketing_source: "organic" (categorical - không normalize)
```

### Ví Dụ 3: High Value User
```
user_id: U00124
gmv_2024: 2.43 (normalized)
  → GMV rất cao (top customer - high value)
  
orders_2024: 2.08 (normalized)
  → Số lượng orders rất cao
  
churn_label: 0
  → Không churn (loyal customer)
```

---

## 4. Thống Kê Distribution

Sau khi normalize, các giá trị sẽ có:
- **Mean ≈ 0**: Giá trị trung bình gần 0
- **Std ≈ 1**: Độ lệch chuẩn gần 1
- **Range**: Thường từ -3 đến +3 (tùy thuộc vào dữ liệu)

### Query để Kiểm Tra
```sql
SELECT 
    AVG(age) as avg_age_norm,
    STDDEV(age) as std_age_norm,
    MIN(age) as min_age_norm,
    MAX(age) as max_age_norm
FROM churn_prediction.users_processed;
```

**Kết quả mong đợi**:
- `avg_age_norm` ≈ 0.000000
- `std_age_norm` ≈ 1.000000
- `min_age_norm` và `max_age_norm` thường trong khoảng [-3, 3]

---

## 5. Cách Đọc Giá Trị trong pgAdmin4

### 5.1. Khi Xem Bảng `users_processed`

**Các cột Numerical (đã normalize)**:
- `age`, `reg_days`, `sessions_30d`, `sessions_90d`
- `orders_30d`, `orders_90d`, `gmv_2024`, `aov_2024`
- Tất cả các cột số khác

→ **Đọc theo quy tắc ở mục 2**

**Các cột Categorical (không normalize)**:
- `user_id`, `country`, `city`, `marketing_source`, `app_version_major`

→ **Đọc giá trị gốc trực tiếp**

### 5.2. Query với Giải Thích

```sql
SELECT 
    user_id,
    age,
    CASE 
        WHEN age < -2 THEN 'Rat tre (very young)'
        WHEN age < -1 THEN 'Tre hon trung binh (younger)'
        WHEN age < 0 THEN 'Hoi tre (slightly young)'
        WHEN age = 0 THEN 'Trung binh (average)'
        WHEN age <= 1 THEN 'Hoi gia (slightly old)'
        WHEN age <= 2 THEN 'Gia hon trung binh (older)'
        ELSE 'Rat gia (very old)'
    END as age_interpretation,
    sessions_30d,
    CASE 
        WHEN sessions_30d < -1 THEN 'Rat it sessions (very inactive)'
        WHEN sessions_30d < 0 THEN 'It sessions (inactive)'
        WHEN sessions_30d <= 1 THEN 'Trung binh den cao (average to high)'
        ELSE 'Rat nhieu sessions (very active)'
    END as sessions_interpretation,
    gmv_2024,
    CASE 
        WHEN gmv_2024 < -1 THEN 'GMV rat thap (low value)'
        WHEN gmv_2024 < 0 THEN 'GMV thap (below average)'
        WHEN gmv_2024 <= 1 THEN 'GMV trung binh den cao (average to high)'
        ELSE 'GMV rat cao (high value customer)'
    END as gmv_interpretation
FROM churn_prediction.users_processed
LIMIT 10;
```

---

## 6. So Sánh Dữ Liệu Gốc vs Normalized

### Nếu Cần Xem Giá Trị Gốc

**Cách 1: Đọc từ file processed parquet**
```python
import pandas as pd
df = pd.read_parquet('data/processed/churn_processed_20251215.parquet')
# Dữ liệu trong file này có thể chứa cả giá trị gốc và normalized
```

**Cách 2: Đọc từ file raw parquet**
```python
df = pd.read_parquet('data/raw/churn_data_20251215.parquet')
# Dữ liệu gốc chưa qua normalization
```

**Cách 3: Đọc từ CSV gốc**
```python
df = pd.read_csv('data/raw/churn_dataset.csv')
# Dữ liệu gốc hoàn toàn
```

---

## 7. Lưu Ý Quan Trọng

### ✅ Những Điều Cần Nhớ

1. **Dữ liệu trong `users_processed` đã được normalize** để phục vụ ML model
2. **Các cột categorical không bị normalize** - đọc giá trị gốc trực tiếp
3. **Các cột numerical đã bị normalize** - đọc theo quy tắc ở mục 2
4. **Giá trị normalized không có đơn vị** - chỉ là số tương đối so với trung bình

### ⚠️ Cảnh Báo

- **Không so sánh trực tiếp** giá trị normalized với giá trị gốc
- **Không cộng/trừ** giá trị normalized với giá trị gốc
- **Chỉ so sánh** các giá trị normalized với nhau (tương đối)

---

## 8. Bảng Tóm Tắt

| Giá Trị Normalized | Ý Nghĩa | Ví Dụ |
|-------------------|---------|-------|
| < -2 | Rất thấp (very low) | `sessions_30d = -2.5` → Rất ít sessions |
| -2 đến -1 | Thấp hơn trung bình (below average) | `age = -1.5` → Trẻ hơn trung bình |
| -1 đến 0 | Hơi thấp (slightly below) | `gmv_2024 = -0.5` → GMV hơi thấp |
| 0 | Trung bình (average) | `age = 0` → Tuổi trung bình |
| 0 đến 1 | Hơi cao (slightly above) | `sessions_30d = 0.5` → Nhiều sessions hơn một chút |
| 1 đến 2 | Cao hơn trung bình (above average) | `gmv_2024 = 1.5` → GMV cao |
| > 2 | Rất cao (very high) | `gmv_2024 = 2.5` → GMV rất cao (top customer) |

---

## 9. Câu Hỏi Thường Gặp (FAQ)

### Q1: Tại sao giá trị age lại là số âm?
**A**: Vì đã được normalize. Giá trị âm nghĩa là tuổi nhỏ hơn trung bình. Để biết tuổi thực tế, cần đọc từ file gốc.

### Q2: Làm sao biết giá trị gốc là bao nhiêu?
**A**: Đọc từ file processed parquet hoặc raw parquet/CSV. Dữ liệu trong PostgreSQL đã được normalize để phục vụ ML model.

### Q3: Có thể reverse normalization không?
**A**: Có, nhưng cần biết mean và std ban đầu. Thông thường không cần thiết vì:
- Model đã được train với dữ liệu normalized
- Predictions cũng dựa trên dữ liệu normalized
- Chỉ cần hiểu ý nghĩa tương đối là đủ

### Q4: Khi nào cần dùng giá trị gốc?
**A**: Khi cần:
- Báo cáo business (GMV thực tế, số orders thực tế)
- Phân tích descriptive (tuổi trung bình thực tế)
- Validation với stakeholders

---

## 10. Tài Liệu Tham Khảo

- **StandardScaler**: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- **Z-score Normalization**: https://en.wikipedia.org/wiki/Standard_score
- **Feature Scaling**: https://en.wikipedia.org/wiki/Feature_scaling

---

**Tác giả**: SkilioPay Data Team  
**Ngày cập nhật**: 2025-12-15  
**Phiên bản**: 1.0

