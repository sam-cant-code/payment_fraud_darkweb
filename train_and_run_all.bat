@echo off
REM ======================================================================
REM == ENHANCED WORKFLOW: Train Model, Run Simulation, & Start Dashboard ==
REM ======================================================================
echo.
echo [STEP 1/3] Training new model with enhanced dataset...
echo ======================================================================
python -m backend.scripts.initialize_and_train
if errorlevel 1 (
    echo [ERROR] Model training failed!
    pause
    exit /b 1
)
echo.
echo [SUCCESS] Model training complete.

REM --- Get the timestamp of the new model ---
set /p LATEST_TIMESTAMP=<models\latest_model_timestamp.txt
echo [INFO] New model timestamp: %LATEST_TIMESTAMP%
echo.

echo [STEP 2/3] Running simulation on all 10,000 transactions...
echo ======================================================================
python -m backend.scripts.run_simulation %LATEST_TIMESTAMP%
if errorlevel 1 (
    echo [ERROR] Simulation run failed!
    pause
    exit /b 1
)
echo.
echo [SUCCESS] Simulation complete. Results saved to output/
echo.

echo [STEP 3/3] Launching Analytics Dashboard...
echo ======================================================================
echo Press Ctrl+C in this window to stop the dashboard.
streamlit run backend/analytics_dashboard.py -- %LATEST_TIMESTAMP%
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