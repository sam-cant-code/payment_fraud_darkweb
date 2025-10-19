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
    pause
    exit /b 1
)

echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

ECHO [2/4] Initializing database and training model...
call python scripts/initialize_and_train.py

ECHO [3/4] Running simulation...
call python scripts/run_simulation.py

ECHO [4/4] Calculating accuracy...
call python scripts/calculate_accuracy.py

ECHO All scripts finished.
pause