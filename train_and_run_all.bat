@echo off
ECHO =======================================================
ECHO (1/3) TRAINING NEW MODEL...
ECHO =======================================================
cd backend/scripts
python initialize_and_train.py

:: Check the correct path: up two levels from 'scripts' to the project root, then into 'models'.
IF NOT EXIST ../../models/latest_model_timestamp.txt (
    ECHO.
    ECHO ❌ ERROR: Model training failed or latest_model_timestamp.txt was not created.
    cd ../..
    pause
    goto :eof
)

:: *** ROBUST FIX ***
:: Use a FOR loop to read the timestamp instead of input redirection (<)
ECHO.
ECHO Reading timestamp...
FOR /F "usebackq tokens=*" %%a IN ("../../models/latest_model_timestamp.txt") DO (
    SET "LATEST_TIMESTAMP=%%a"
)

IF NOT DEFINED LATEST_TIMESTAMP (
    ECHO.
    ECHO ❌ ERROR: Could not read timestamp from latest_model_timestamp.txt.
    cd ../..
    pause
    goto :eof
)

ECHO.
ECHO =======================================================
ECHO (2/3) RUNNING SIMULATION FOR MODEL: %LATEST_TIMESTAMP%
ECHO =======================================================
python run_simulation.py %LATEST_TIMESTAMP%

ECHO.
ECHO =======================================================
ECHO (3/3) CALCULATING ACCURACY FOR MODEL: %LATEST_TIMESTAMP%
ECHO =======================================================
python calculate_accuracy.py %LATEST_TIMESTAMP%

cd ../..
ECHO.
ECHO =======================================================
ECHO ✓ WORKFLOW COMPLETE
ECHO =======================================================
pause