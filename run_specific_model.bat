@echo off
REM ======================================================================
REM == Run Simulation & View Dashboard for a SPECIFIC Model Timestamp ==
REM ======================================================================

set MODEL_TIMESTAMP=%1

if "%MODEL_TIMESTAMP%"=="" (
    echo [ERROR] No timestamp provided.
    echo.
    echo Usage: run_specific_model.bat ^<TIMESTAMP^>
    echo Example: run_specific_model.bat 20251020_210357
    pause
    exit /b 1
)

echo [INFO] Running workflow for model: %MODEL_TIMESTAMP%
echo.

echo [STEP 1/2] Running simulation on all 10,000 transactions...
echo ======================================================================
python -m backend.scripts.run_simulation %MODEL_TIMESTAMP%
if errorlevel 1 (
    echo [ERROR] Simulation run failed!
    pause
    exit /b 1
)
echo.
echo [SUCCESS] Simulation complete. Results saved to output/
echo.

echo [STEP 2/2] Launching Analytics Dashboard for model %MODEL_TIMESTAMP%...
echo ======================================================================
echo Press Ctrl+C in this window to stop the dashboard.
streamlit run backend/analytics_dashboard.py -- %MODEL_TIMESTAMP%
if errorlevel 1 (
    echo [ERROR] Failed to start Streamlit dashboard.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo WORKFLOW COMPLETE
echo ======================================================================
echo.
pause