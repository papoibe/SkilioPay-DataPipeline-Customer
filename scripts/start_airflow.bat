@echo off
echo [INFO] Starting Airflow Services...
docker-compose -f docker-compose.airflow.yml up -d

echo [INFO] Airflow is running!
echo [INFO] Access Web UI at: http://localhost:8080
echo [INFO] User/Pass: airflow / airflow
pause
