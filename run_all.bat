@echo off
echo ========================================================
echo Launching DeepFake AI Fullstack Engine
echo ========================================================
start "DeepFake Backend (Port 8000)" cmd /k "%~dp0run_backend.bat"
start "DeepFake Frontend (Port 5173)" cmd /k "%~dp0run_frontend.bat"
echo Both servers started in dedicated windows!
echo Frontend will be accessible at: http://localhost:5173
echo Backend API and Swagger docs at: http://127.0.0.1:8000/docs
