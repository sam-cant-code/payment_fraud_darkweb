@echo off
REM ============================================================
REM Starts the Backend API Server
REM ============================================================
echo ============================================================
echo STARTING BACKEND API SERVER...
echo ============================================================
echo.

REM Navigate to backend folder
cd backend
if errorlevel 1 (
    echo [ERROR] 'backend' directory not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

if not exist "venv" (
    echo [ERROR] Virtual environment 'venv' not found.
    echo Please run 'setup.bat' first from the root directory.
    cd ..
    pause
    exit /b 1
)

if not exist "server.py" (
    echo [ERROR] 'server.py' not found in backend directory!
    cd ..
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    cd ..
    pause
    exit /b 1
)
echo.

echo ============================================================
echo Starting API server at http://127.0.0.1:5000
echo ============================================================
echo.
echo The server will now:
echo  - Serve API endpoints at /api/*
echo  - Monitor blockchain transactions (if Alchemy URL configured)
echo  - Process transactions through AI agents
echo  - Save results to output/all_processed_transactions.csv
echo.
echo Important:
echo  - Make sure you ran 'setup.bat' first
echo  - Configure ALCHEMY_HTTP_URL in backend/.env for live monitoring
echo  - Frontend will connect to this server on port 5000
echo.
echo Press Ctrl+C to stop the server.
echo ============================================================
echo.

python server.py

REM Return to root directory when server stops
cd ..
pause