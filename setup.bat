@echo off
REM ============================================================
REM Blockchain Fraud Detection MVP - Automated Setup Script
REM For Windows
REM ============================================================

echo ============================================================
echo BLOCKCHAIN FRAUD DETECTION MVP - AUTOMATED SETUP
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/8] Python found: 
python --version
echo.

REM Create project directory structure
echo [2/8] Creating directory structure...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "output" mkdir output
echo     - Created: data/
echo     - Created: models/
echo     - Created: output/
echo.

REM Create virtual environment
echo [3/8] Creating virtual environment...
if exist "venv" (
    echo     Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    echo     - Created: venv/
)
echo.

REM Activate virtual environment and install packages
echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo     - Virtual environment activated
echo.

REM Upgrade pip
echo [5/8] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo     - pip upgraded
echo.

REM Create requirements.txt if it doesn't exist
echo [6/8] Checking requirements.txt...
if not exist "requirements.txt" (
    echo Creating requirements.txt...
    (
        echo pandas^>=1.3.0
        echo numpy^>=1.21.0
        echo scikit-learn^>=1.0.0
        echo streamlit^>=1.10.0
    ) > requirements.txt
    echo     - Created: requirements.txt
) else (
    echo     - requirements.txt already exists
)
echo.

REM Install dependencies
echo [7/8] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo     - All dependencies installed successfully
echo.

REM Check if Python files exist
echo [8/8] Checking Python files...
set FILES_MISSING=0

if not exist "agents.py" (
    echo     [MISSING] agents.py
    set FILES_MISSING=1
)
if not exist "0_setup_database.py" (
    echo     [MISSING] 0_setup_database.py
    set FILES_MISSING=1
)
if not exist "1_run_simulation.py" (
    echo     [MISSING] 1_run_simulation.py
    set FILES_MISSING=1
)
if not exist "2_dashboard.py" (
    echo     [MISSING] 2_dashboard.py
    set FILES_MISSING=1
)

if %FILES_MISSING%==1 (
    echo.
    echo [WARNING] Some Python files are missing!
    echo Please create the following files and copy the code from the artifacts:
    echo   - agents.py
    echo   - 0_setup_database.py
    echo   - 1_run_simulation.py
    echo   - 2_dashboard.py
) else (
    echo     - All Python files found
)
echo.

REM Display completion message
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Virtual environment is ACTIVE (you should see 'venv' in prompt)
echo.
echo Next steps:
if %FILES_MISSING%==1 (
    echo 1. Create the missing Python files ^(see warning above^)
    echo 2. Run: python 0_setup_database.py
    echo 3. Run: python 1_run_simulation.py
    echo 4. Run: streamlit run 2_dashboard.py
) else (
    echo 1. Run: python 0_setup_database.py
    echo 2. Run: python 1_run_simulation.py
    echo 3. Run: streamlit run 2_dashboard.py
)
echo.
echo To deactivate virtual environment later, run: deactivate
echo To activate again, run: venv\Scripts\activate
echo ============================================================
echo.

REM Keep window open
echo Press any key to exit...
pause >nul