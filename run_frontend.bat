@echo off
echo Starting DeepFake Detection Frontend (Vite on Port 5173)...
cd /d %~dp0app\frontend
npm run dev
pause
