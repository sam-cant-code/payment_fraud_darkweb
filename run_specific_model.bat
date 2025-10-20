@echo off
set TIMESTAMP=%1

IF "%TIMESTAMP%"=="" (
    ECHO ❌ ERROR: No timestamp provided.
    ECHO.
    ECHO   Usage: run_specific_model.bat <TIMESTAMP>
    ECHO   Example: run_specific_model.bat 20251019_223000
    pause
    goto :eof
)

ECHO =======================================================
ECHO (1/2) RUNNING SIMULATION FOR MODEL: %TIMESTAMP%
ECHO =======================================================
cd backend/scripts
python run_simulation.py %TIMESTAMP%

ECHO.
ECHO =======================================================
ECHO (2/2) CALCULATING ACCURACY FOR MODEL: %TIMESTAMP%
ECHO =======================================================
python calculate_accuracy.py %TIMESTAMP%

cd ../..
ECHO.
ECHO =======================================================
ECHO ✓ WORKFLOW COMPLETE
ECHO =======================================================
pause