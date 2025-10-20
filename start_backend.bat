@echo off
REM ============================================================
REM Starts the Backend Flask Server
REM ============================================================
echo ============================================================
echo STARTING BACKEND SERVER
echo ============================================================
echo.

REM --- This script assumes it's run from the ROOT folder ---

cd backend
if errorlevel 1 (
    echo [ERROR] 'backend' directory not found!
    echo Please run this script from the project's root directory.
    pause
    exit /b 1
)

if not exist "venv" (
    echo [ERROR] Virtual environment 'venv' not found.
    echo Please run 'setup.bat' first.
    cd ..
    pause
    exit /b 1
)

echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    cd ..
    pause
    exit /b 1
)
echo.

REM --- CORRECTED CHECK: Look for the timestamp file ---
if not exist "..\models\latest_model_timestamp.txt" (
    echo [WARNING] Latest model timestamp file not found!
    echo Please run 'train_and_run_all.bat' first to train a model.
    cd ..
    pause
    exit /b 1
)

echo [2/2] Starting Flask server...
echo      (Will load the latest model automatically)
echo      (Press CTRL+C to stop)
echo.
python server.py
if errorlevel 1 (
    echo [ERROR] Failed to start Flask server.
    cd ..
    pause
    exit /b 1
)

echo Server stopped.
cd ..
pause