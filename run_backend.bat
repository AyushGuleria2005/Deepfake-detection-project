@echo off
echo Starting DeepFake Detection Backend (FastAPI on Port 8000)...
cd /d %~dp0
call .\venv\Scripts\activate.bat
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
