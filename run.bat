@echo off
REM ============================================================
REM Blockchain Fraud Detection MVP - Quick Run Script
REM For Windows
REM ============================================================

echo ============================================================
echo BLOCKCHAIN FRAUD DETECTION MVP - QUICK RUN
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup_project.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Check if setup has been run
if not exist "data\mock_blockchain_transactions.json" (
    echo ============================================================
    echo STEP 1: RUNNING SETUP
    echo ============================================================
    echo.
    python 0_setup_database.py
    if errorlevel 1 (
        echo [ERROR] Setup failed
        pause
        exit /b 1
    )
    echo.
)

REM Run simulation
echo ============================================================
echo STEP 2: RUNNING SIMULATION
echo ============================================================
echo.
python 1_run_simulation.py
if errorlevel 1 (
    echo [ERROR] Simulation failed
    pause
    exit /b 1
)
echo.

REM Launch dashboard
echo ============================================================
echo STEP 3: LAUNCHING DASHBOARD
echo ============================================================
echo.
echo Dashboard will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the dashboard when done
echo.
streamlit run 2_dashboard.py

pause