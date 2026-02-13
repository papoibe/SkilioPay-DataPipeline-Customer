# Script khởi động Airflow
Write-Host "Khoi dong Airflow..." -ForegroundColor Cyan

# Active venv
.\.venv\Scripts\Activate.ps1

# Kiểm tra DAG
Write-Host "`nKiem tra DAG..." -ForegroundColor Yellow
airflow dags list | Select-String "skilio"

# Khởi động Webserver (Terminal 1)
Write-Host "`n[1] Khoi dong Airflow Webserver (Terminal 1)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\.venv\Scripts\Activate.ps1; Write-Host 'Airflow Webserver - http://localhost:8080' -ForegroundColor Cyan; airflow webserver --port 8080"

Start-Sleep -Seconds 3

# Khởi động Scheduler (Terminal 2)
Write-Host "[2] Khoi dong Airflow Scheduler (Terminal 2)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\.venv\Scripts\Activate.ps1; Write-Host 'Airflow Scheduler' -ForegroundColor Cyan; airflow scheduler"

Write-Host "`n[OK] Airflow da khoi dong!" -ForegroundColor Green
Write-Host "`nTruy cap Airflow UI: http://localhost:8080" -ForegroundColor Yellow
Write-Host "DAG name: skilio_pay_churn_prediction_pipeline" -ForegroundColor Yellow
Write-Host "`nDe chay DAG:" -ForegroundColor Cyan
Write-Host "1. Mo http://localhost:8080" -ForegroundColor White
Write-Host "2. Tim DAG 'skilio_pay_churn_prediction_pipeline'" -ForegroundColor White
Write-Host "3. Bat DAG (toggle switch)" -ForegroundColor White
Write-Host "4. Click nut Play de trigger manually" -ForegroundColor White

