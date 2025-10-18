"""
Database Management Module
Handles schema creation and connection for the unified SQLite database.
"""

import sqlite3
import os

DB_FILE = 'data/app.db'

def get_db_connection():
    """Establishes a connection to the database."""
    os.makedirs('data', exist_ok=True) # Ensure data directory exists
    conn = sqlite3.connect(DB_FILE)
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
    
    # --- Wallet Historical Profiles ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS wallet_profiles (
        wallet_address TEXT PRIMARY KEY,
        avg_transaction_value REAL,
        transaction_count INTEGER,
        last_updated TEXT
    )
    """)
    
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
    print("Database schema created/verified successfully.")

if __name__ == '__main__':
    # This allows running 'python database.py' to initialize the DB
    create_schema()
    print(f"Database file created at {DB_FILE}")