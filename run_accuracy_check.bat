@echo off
REM ============================================================
REM Runs the full Setup -> Simulate -> Accuracy Check workflow.
REM This single file automates the entire testing process.
REM Make sure Python and required packages are installed (run setup.bat once first).
REM ============================================================

echo ============================================================
echo RUNNING FULL ACCURACY CHECK WORKFLOW
echo ============================================================
echo.

REM Navigate into the backend directory
cd backend
if errorlevel 1 (
    echo [ERROR] 'backend' directory not found!
    echo Please run this script from the project's root directory.
    pause
    exit /b 1
)
echo [1/6] In 'backend' directory.
echo.

REM Check for virtual environment
if not exist "venv" (
    echo [ERROR] Virtual environment 'venv' not found.
    echo Please run 'setup.bat' in the root directory once to create it.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo.

REM Delete previous output and mock data files for a clean run
echo [3/6] Deleting previous output and data files (if they exist)...
if exist "output\all_processed_transactions.csv" (
    del "output\all_processed_transactions.csv"
    echo     - Deleted output\all_processed_transactions.csv
) else (
    echo     - No previous output file found.
)
if exist "data\mock_blockchain_transactions.json" (
    del "data\mock_blockchain_transactions.json"
    echo     - Deleted data\mock_blockchain_transactions.json
) else (
    echo     - No previous mock data file found.
)
echo.


REM Run the setup script to generate fresh, labeled mock data
echo [4/6] Running setup (0_setup_database.py) to generate new mock data...
REM Redirect stdout to NUL to prevent its output from interfering with the batch script
python 0_setup_database.py > NUL
if errorlevel 1 (
    echo [ERROR] Failed to run setup script (0_setup_database.py).
    call venv\Scripts\deactivate.bat
    pause
    exit /b 1
)
echo     - Setup script completed successfully.
echo.

REM Run the simulation script to process the mock data
echo [5/6] Running simulation (1_run_simulation.py) to process the data...
REM Redirect stdout to NUL
python 1_run_simulation.py > NUL
if errorlevel 1 (
    echo [ERROR] Failed to run simulation script (1_run_simulation.py).
    call venv\Scripts\deactivate.bat
    pause
    exit /b 1
)
echo     - Simulation script completed successfully.
echo.

REM Run the accuracy calculation script to get the results
echo [6/6] Calculating accuracy (3_calculate_accuracy.py)...
REM This script's output IS the result, so we DO NOT redirect it.
python 3_calculate_accuracy.py
if errorlevel 1 (
    echo [ERROR] Failed to run accuracy script (3_calculate_accuracy.py).
    call venv\Scripts\deactivate.bat
    pause
    exit /b 1
)
echo.

REM Deactivate environment (optional but good practice)
echo Deactivating virtual environment...
call venv\Scripts\deactivate.bat
echo.

echo ============================================================
echo ACCURACY CHECK WORKFLOW COMPLETE!
echo The results are displayed above.
echo ============================================================
echo.
pause

