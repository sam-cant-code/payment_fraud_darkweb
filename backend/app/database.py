"""
Database Management Module
Handles schema creation and connection for the unified SQLite database.

MODIFIED: Removed 'wallet_profiles' table as it's no longer needed.
Live data is fetched by Agent_2 from Etherscan.
"""

import sqlite3
import os
from pathlib import Path

# --- PATH CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent  # backend/app/
BACKEND_DIR = SCRIPT_DIR.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # project root

# *** MODIFICATION: Use PROJECT_ROOT for DB path ***
DB_FILE = PROJECT_ROOT / 'data' / 'app.db'

def get_db_connection():
    """Establishes a connection to the database."""
    os.makedirs(DB_FILE.parent, exist_ok=True) # Ensure data directory exists
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    return conn

def create_schema():
    """Creates all necessary tables in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # --- Threat Intelligence Table ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS threat_list (
        wallet_address TEXT PRIMARY KEY,
        added_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- Wallet Historical Profiles (REMOVED) ---
    # This table is no longer created as Agent_2 uses the live Etherscan API.
    # We add a DROP TABLE command to clean up the old 'fake' table if it exists.
    cursor.execute("DROP TABLE IF EXISTS wallet_profiles")

    # --- All Transactions Table ---
    # Replaces 'flagged_transactions'. We now store ALL results.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS all_transactions (
        tx_hash TEXT PRIMARY KEY,
        from_address TEXT,
        to_address TEXT,
        value_eth REAL,
        gas_price REAL,
        timestamp TEXT,
        
        -- Agent results
        final_status TEXT,
        final_score REAL,
        reasons TEXT,
        agent_1_score REAL,
        agent_2_score REAL,
        agent_3_score REAL,
        ml_probability REAL,
        
        -- Ground truth label
        is_fraud INTEGER, 
        
        -- Analyst review
        analyst_status TEXT DEFAULT 'PENDING', -- PENDING, APPROVED, DENIED
        reviewed_by TEXT,
        reviewed_on TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
    print("Database schema created/verified successfully. (Removed wallet_profiles table)")

if __name__ == '__main__':
    # This allows running 'python -m app.database' from 'backend' folder
    create_schema()
    print(f"Database file created at {DB_FILE}")