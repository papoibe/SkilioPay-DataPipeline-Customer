@echo off
echo [INFO] Building Airflow Docker Image...
docker-compose -f docker-compose.airflow.yml build

echo [INFO] Starting Postgres & Redis...
docker-compose -f docker-compose.airflow.yml up -d postgres redis

echo [INFO] Waiting for Database to be ready...
timeout /t 10

echo [INFO] Initializing Airflow Database...
docker-compose -f docker-compose.airflow.yml run --rm airflow-webserver airflow db init

echo [INFO] Creating Admin User...
docker-compose -f docker-compose.airflow.yml run --rm airflow-webserver airflow users create ^
    --username airflow ^
    --firstname Thinh ^
    --lastname Lee_Nguyen_Phuoc ^
    --role Admin ^
    --email admin@skiliopay.com ^
    --password airflow

echo [INFO] Cleaning up...
docker-compose -f docker-compose.airflow.yml down

echo [SUCCESS] Airflow initialized successfully!
echo [INFO] You can now run scripts\start_airflow.bat
pause
