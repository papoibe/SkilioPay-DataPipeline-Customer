# Hướng Dẫn Chạy Airflow DAG

## 1. Khởi Động Airflow

### Bước 1: Khởi tạo Airflow Database (nếu chưa có)
```bash
# Active venv
.\.venv\Scripts\Activate.ps1

# Khởi tạo database
airflow db init
```

### Bước 2: Tạo User Admin (nếu chưa có)
```bash
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### Bước 3: Khởi động Airflow Webserver
```bash
# Terminal 1: Webserver
airflow webserver --port 8080
```

### Bước 4: Khởi động Airflow Scheduler
```bash
# Terminal 2: Scheduler
airflow scheduler
```

## 2. Truy Cập Airflow UI

Mở trình duyệt và truy cập:
```
http://localhost:8080
```

Đăng nhập với:
- Username: `admin`
- Password: `admin` (hoặc password bạn đã tạo)

## 3. Chạy DAG

1. Trong Airflow UI, tìm DAG `skilio_pay_churn_prediction_pipeline`
2. Bật DAG (toggle switch bên trái)
3. Click vào DAG name để xem chi tiết
4. Click nút "Play" (▶) để trigger DAG manually
5. Hoặc đợi schedule tự động chạy (daily)

## 4. Xem Logs và Kết Quả

- Click vào task để xem logs
- Xem kết quả trong các bảng:
  - `churn_prediction.users_processed`
  - `churn_prediction.features`
- Model mới sẽ được lưu trong `models/`

## 5. Cấu Hình DAG

DAG được cấu hình trong `dags/churn_prediction_pipeline.py`:
- **Schedule**: `@daily` (chạy hàng ngày)
- **Start Date**: 2024-01-01
- **Owner**: Le Nguyen Phuoc Thinh
- **Email**: lenguyenphuocthinh1234@gmail.com

## 6. Troubleshooting

### Lỗi: DAG không hiển thị
- Kiểm tra file DAG có trong thư mục `dags/`
- Kiểm tra syntax Python có đúng không
- Xem logs của scheduler

### Lỗi: Import module không được
- Đảm bảo `src/` trong PYTHONPATH
- Kiểm tra dependencies đã cài đặt chưa

### Lỗi: Database connection
- Kiểm tra PostgreSQL đang chạy
- Kiểm tra config trong `config/config.yaml`

## 7. Script Chạy Nhanh

Tạo file `start_airflow.ps1`:
```powershell
# Start Airflow
.\.venv\Scripts\Activate.ps1

# Terminal 1: Webserver
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\PipelineSkilioPayCustomer; .\.venv\Scripts\Activate.ps1; airflow webserver --port 8080"

# Terminal 2: Scheduler  
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\PipelineSkilioPayCustomer; .\.venv\Scripts\Activate.ps1; airflow scheduler"
```

