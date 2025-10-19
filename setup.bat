@echo off
REM ============================================================
REM Setup Script for Backend (Python)
REM ============================================================

echo ============================================================
echo SETTING UP PYTHON BACKEND
echo ============================================================
echo.

REM Navigate into the backend directory
cd backend
if errorlevel 1 (
    echo [ERROR] 'backend' directory not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

echo [1/5] Now in 'backend' directory.
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    cd ..
    pause
    exit /b 1
)
echo [2/5] Python found: 
python --version
echo.

REM Create virtual environment
echo [3/5] Creating virtual environment 'venv'...
if exist "venv" (
    echo     Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        cd ..
        pause
        exit /b 1
    )
    echo     - Created: venv/
)
echo.

REM Activate virtual environment and install packages
echo [4/5] Activating virtual environment and installing packages...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    cd ..
    pause
    exit /b 1
)

echo     - Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Pip upgrade failed, continuing anyway...
)

echo     - Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Please check your internet connection and requirements.txt file
    cd ..
    pause
    exit /b 1
)
echo     - All backend dependencies installed.
echo.

REM Run the one-time database and model setup
echo [5/5] Running initial database and model setup...
python scripts/initialize_and_train.py
if errorlevel 1 (
    echo [ERROR] Failed to initialize database and train model
    echo Please check the error messages above
    cd ..
    pause
    exit /b 1
)
echo.

cd ..

echo ============================================================
echo BACKEND SETUP COMPLETE!
echo ============================================================
echo.
echo What was created:
echo  - Virtual environment (backend/venv/)
echo  - Database (backend/data/wallet_profiles.db)
echo  - Trained ML model (backend/models/fraud_model.pkl)
echo  - Threat list (backend/data/dark_web_wallets.txt)
echo  - Mock transaction data (backend/data/mock_blockchain_transactions.json)
echo.
echo Next steps:
echo  1. [OPTIONAL] Run 'npm install' in the 'frontend' directory
echo  2. Run 'start_backend.bat' to start the API server
echo  3. Run 'run_accuracy_check.bat' to test the system
echo  4. [OPTIONAL] In a NEW terminal, run 'npm run dev' in frontend
echo.
pause