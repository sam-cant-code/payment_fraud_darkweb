@echo off
REM ============================================================
REM V4.0 Setup Script for Backend (Python)
REM ============================================================

echo ============================================================
echo SETTING UP V4.0 PYTHON BACKEND
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
echo [4/5] Activating virtual environment and installing V4 packages...
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
echo     - All V4 backend dependencies installed.
echo.

REM Run the one-time database and V4 model setup
echo [5/5] Running initial V4 database and model training...
echo (This may take several minutes)
python scripts/initialize_and_train.py
if errorlevel 1 (
    echo [ERROR] Failed to initialize database and train V4 model
    echo Please check the error messages above
    cd ..
    pause
    exit /b 1
)
echo.
cd ..

echo ============================================================
echo V4.0 BACKEND SETUP COMPLETE!
echo ============================================================
echo.
echo What was created:
echo  - Virtual environment (backend/venv/)
echo  - V4 Database (backend/data/wallet_profiles.db)
echo  - V4 Trained ML Ensemble (backend/models/fraud_model_...pkl)
echo  - V4 Model Scaler (backend/models/scaler_...pkl)
echo  - V4 Model Metadata (backend/models/model_metadata_...json)
echo  - Latest model pointer (backend/models/latest_model_timestamp.txt)
echo  - Threat list (backend/data/dark_web_wallets.txt)
echo  - V4 Mock transaction data (backend/data/mock_blockchain_transactions.json)
echo.
echo Next steps:
echo  1. Run 'start_backend.bat' to start the API server
echo  2. (Optional) Run 'python backend/scripts/evaluate_model.py' for plots
echo  3. (Optional) Run 'npm install' then 'npm run dev' in frontend/
echo.
pause