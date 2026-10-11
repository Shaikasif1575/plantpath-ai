@echo off
title PlantPath AI - Starting...
color 0A
cd /d "%~dp0"

echo ============================================
echo        PlantPath AI - Local Launcher
echo ============================================
echo.

REM ---------- Backend ----------
echo [1/3] Starting backend on port 8000 ...
set DEMO_MODE=true
set MODEL_PATH=models\plantpath_model.keras
set CLASS_NAMES_PATH=data\class_names.json
set DISEASES_PATH=data\diseases.json
set MODEL_METRICS_PATH=data\model_metrics.json
set DB_PATH=backend\app\database\plantpath.db
set MAX_UPLOAD_SIZE_MB=10
set CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

start "PlantPath Backend :8000" /min cmd /c "cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

REM ---------- Frontend ----------
echo [2/3] Starting frontend on port 3000 ...
if not exist "frontend\node_modules" (
    echo       Installing frontend dependencies...
    cd frontend && call npm install && cd ..
)
start "PlantPath Frontend :3000" /min cmd /c "cd frontend && npm run start"

REM ---------- Wait & open ----------
echo [3/3] Waiting for servers to boot...
timeout /t 8 /nobreak >nul

echo.
echo ============================================
echo   OPEN IN YOUR BROWSER:
echo
echo      http://localhost:3000
echo
echo   API health : http://localhost:8000/api/health
echo   API docs   : http://localhost:8000/docs
echo ============================================
echo.
echo   Close the two minimized windows titled
echo   "PlantPath Backend" and "PlantPath Frontend"
echo   to stop the servers.
echo.

start "" "http://localhost:3000"
pause
