@echo off
REM ============================================================
REM Start Flask Backend Server
REM ============================================================
echo ============================================================
echo STARTING BACKEND SERVER
echo ============================================================
echo.

cd backend
if errorlevel 1 (
    echo [ERROR] 'backend' directory not found!
    pause
    exit /b 1
)

if not exist "venv" (
    echo [ERROR] Virtual environment not found!
    echo Please run 'setup.bat' first
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

REM Check if required files exist
if not exist "models\fraud_model.pkl" (
    echo [WARNING] Model files not found!
    echo Please run 'setup.bat' first to train the model
    echo.
    pause
    exit /b 1
)

echo [2/2] Starting Flask server on http://localhost:5000
echo.
echo ============================================================
echo SERVER RUNNING
echo ============================================================
echo.
echo API Endpoints:
echo  - http://localhost:5000/api/status
echo  - http://localhost:5000/api/flagged-transactions
echo  - http://localhost:5000/api/threats
echo.
echo WebSocket:
echo  - ws://localhost:5000/socket.io
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python server.py

cd ..