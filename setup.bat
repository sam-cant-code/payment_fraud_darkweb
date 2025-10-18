@echo off
REM ============================================================
REM NEW Setup Script for Backend (Python/Flask)
REM ============================================================

echo ============================================================
echo SETTING UP PYTHON BACKEND
echo ============================================================
echo.

REM Navigate into the backend directory
cd backend
if errorlevel 1 (
    echo [ERROR] 'backend' directory not found!
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
    echo     - Created: venv/
)
echo.

REM Activate virtual environment and install packages
echo [4/5] Activating virtual environment and installing packages...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo     - Upgrading pip...
python -m pip install --upgrade pip --quiet
echo     - Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo     - All backend dependencies installed.
echo.

REM Run the one-time database setup
echo [5/5] Running initial database and model setup...
python 0_setup_database.py
echo.

echo ============================================================
echo BACKEND SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo 1. Run 'npm install' in the 'frontend' directory.
echo 2. Run 'start_backend.bat' in this root folder.
echo 3. In a NEW terminal, run 'start_frontend.bat' in this root folder.
echo.
pause