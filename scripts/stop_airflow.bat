@echo off
echo [INFO] Stopping Airflow Services...
docker-compose -f docker-compose.airflow.yml down
echo [INFO] Airflow stopped.
pause
