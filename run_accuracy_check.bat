@echo off
ECHO Starting script execution...

ECHO [1/3] Running database setup...
call python "C:\Projects\payment_fraud_darkweb\backend\0_setup_database.py"

ECHO [2/3] Running simulation...
call python "C:\Projects\payment_fraud_darkweb\backend\1_run_simulation.py"

ECHO [3/3] Calculating accuracy...
call python "C:\Projects\payment_fraud_darkweb\backend\3_calculate_accuracy.py"

ECHO All scripts finished.
pause