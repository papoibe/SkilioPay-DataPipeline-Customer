# Hướng Dẫn Vận Hành Hệ Thống SkilioPay Churn Prediction

## 1. Yêu Cầu Hệ Thống

- **Python**: 3.9+ (Active venv: `.\.venv\Scripts\Activate.ps1`)
- **PostgreSQL**: Đang chạy (Port 5432). Database `skilio_pay` đã được tạo.
- **Airflow**: 2.7+ (Đã cài đặt trong venv).
  > **Lưu ý**: Trên Windows, Airflow cần chạy qua **WSL (Windows Subsystem for Linux)** hoặc **Docker** vì module `pwd` không được hỗ trợ trên Windows native.


---

## 2. Cấu Hình Airflow

Trước khi chạy Airflow lần đầu, cần khởi tạo Database và tạo User (nếu chưa làm):

```powershell
# Active venv
.\.venv\Scripts\Activate.ps1

# Khởi tạo DB (chỉ chạy 1 lần)
airflow db init

# Tạo User Admin
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
```

---

## 3. Cách Chạy Hệ Thống

### Cách A: Chạy Thủ Công (Manual) - Dùng khi Test hoặc Dev

Chạy từng script theo thứ tự:

1. **Ingestion**: `python scripts/run_ingestion.py`
2. **Processing**: `python scripts/run_processing.py`
3. **Warehouse**: `python scripts/run_warehouse.py`
4. **Training**: `python scripts/run_training.py`
5. **API**: `uvicorn src.serving.api:app --host 0.0.0.0 --port 8000`

### Cách B: Chạy Tự Động bằng Airflow (Production)

Sử dụng script `start_airflow.ps1` có sẵn:

```powershell
.\start_airflow.ps1
```

Script này sẽ:
1. Kiểm tra DAG `skilio_pay_churn_prediction_pipeline`.
2. Mở 2 cửa sổ terminal mới:
   - **Terminal 1**: Running `airflow webserver`
   - **Terminal 2**: Running `airflow scheduler`

Sau đó:
1. Mở trình duyệt: [http://localhost:8080](http://localhost:8080)
2. Đăng nhập: `admin` / `admin`
3. Tìm DAG `skilio_pay_churn_prediction_pipeline` và bật nút **ON**.
4. Click nút **Play** (Trigger DAG) để chạy ngay lập tức.

---

## 4. Kiểm Tra Dữ Liệu Hàng Ngày

Để kiểm tra xem dữ liệu hôm nay đã được xử lý và load vào Warehouse chưa, chạy lệnh:

```powershell
python scripts/check_daily_data.py
```

Output mong đợi:
```
=== CHECKING DATA FOR 2026-02-11 ===

[OK] Raw Data found: data/raw/churn_data_20260211.parquet
   Rows: 50000

[OK] Processed Data found: data/processed/churn_processed_20260211.parquet
   Rows: 50000
   Cols: 181

[INFO] Checking Warehouse (Postgres)...
[OK] Users Validation: 50000 rows in 'users_processed'
[OK] Features Validation: 50000 rows in 'features'

=== CHECK COMPLETE ===
```

---

## 5. Dashboard

Khởi động Dashboard bằng Streamlit:

```powershell
streamlit run src/dashboard/app.py --server.port 8501
```

Mở trình duyệt: [http://localhost:8501](http://localhost:8501)

Dashboard gồm 5 trang:
1. **🏠 Tổng Quan** — KPIs churn rate, donut chart, phân bố tuổi, RFM
2. **📈 Phân Tích Chi Tiết** — Theo quốc gia, hành vi mua hàng, engagement
3. **🤖 Model Performance** — Accuracy, ROC-AUC, Feature Importance
4. **🔍 Tra Cứu User** — Nhập User ID để xem dự đoán churn
5. **⚙️ Pipeline Status** — Trạng thái pipeline & API health


---

## 6. Vận Hành Airflow với Docker (Khuyên Dùng trên Windows)

Docker giúp tránh các lỗi không tương thích thư viện trên Windows.

### 6.1. Cài đặt lần đầu (Chỉ chạy 1 lần)
Sử dụng script `init_airflow.bat` để build image, khởi tạo database và tạo admin user.

```powershell
.\scripts\init_airflow.bat
```
*   **Username**: `airflow`
*   **Password**: `airflow`

### 6.2. Chạy hàng ngày (Start)
Để bắt đầu làm việc, chạy lệnh sau để bật toàn bộ services (Webserver, Scheduler, Worker):

```powershell
.\scripts\start_airflow.bat
```
*   Services sẽ chạy ngầm (detached mode).
*   Truy cập Airflow UI: [http://localhost:8080](http://localhost:8080)

### 6.3. Dừng hệ thống (Stop)
Khi không sử dụng nữa, chạy lệnh sau để tắt services và giải phóng tài nguyên:

```powershell
.\scripts\stop_airflow.bat
```

> **Lưu ý**: Folder `dags/`, `src/`, `data/`, `logs/` được mount trực tiếp vào container, nên mọi thay đổi code của bạn sẽ được cập nhật ngay lập tức trên Airflow mà không cần rebuild image.