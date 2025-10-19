@echo off
REM ============================================================
REM Starts the Backend API Server
REM ============================================================
echo ============================================================
echo STARTING BACKEND API SERVER...
echo ============================================================
echo.
cd backend
if errorlevel 1 (
    echo [ERROR] 'backend' directory not found!
    pause
    exit /b 1
)

if not exist "venv" (
    echo [ERROR] Virtual environment 'venv' not found.
    echo Please run 'setup.bat' first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Starting API server at http://127.0.0.1:5000
echo Press Ctrl+C to stop the server.
echo.
python server.py

pause