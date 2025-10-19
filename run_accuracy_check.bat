@echo off
REM ============================================================
REM Runs the full simulation and accuracy check
REM ============================================================
echo ============================================================
echo RUNNING ACCURACY CHECK...
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

echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    cd ..
    pause
    exit /b 1
)
echo.

echo [2/4] Initializing database and training model...
python scripts/initialize_and_train.py
if errorlevel 1 (
    echo [ERROR] Failed to initialize and train
    cd ..
    pause
    exit /b 1
)
echo.

echo [3/4] Running simulation...
python scripts/run_simulation.py
if errorlevel 1 (
    echo [ERROR] Failed to run simulation
    cd ..
    pause
    exit /b 1
)
echo.

echo [4/4] Calculating accuracy...
python scripts/calculate_accuracy.py
if errorlevel 1 (
    echo [ERROR] Failed to calculate accuracy
    cd ..
    pause
    exit /b 1
)
echo.

echo ============================================================
echo ALL SCRIPTS FINISHED SUCCESSFULLY!
echo ============================================================
echo.
echo Results saved to: backend/output/all_processed_transactions.csv
echo.
echo You can now:
echo  1. View the dashboard: python backend/analytics_dashboard.py
echo  2. Start the backend: start_backend.bat
echo.

cd ..
pause