"""
Setup Script for Blockchain Fraud Detection MVP
Creates mock data, database, and trains the ML model
"""

import json
import sqlite3
import pickle
import random
import os
from sklearn.ensemble import IsolationForest
import numpy as np
from datetime import datetime # <--- Added this import

def create_directories():
    """Create necessary directories if they don't exist"""
    # ... (function remains the same) ...
    directories = ['data', 'models', 'output']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def create_mock_transactions():
    """Create mock blockchain transactions JSON file"""
    # ... (function remains the same) ...
    print("\n=== Creating Mock Blockchain Transactions ===")
    filepath = 'data/mock_blockchain_transactions.json'
    malicious_wallets = [ "0x1234567890abcdef1234567890abcdef12345678", "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "0x9876543210fedcba9876543210fedcba98765432" ]
    normal_wallets = [ "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "0xcccccccccccccccccccccccccccccccccccccccc", "0xdddddddddddddddddddddddddddddddddddddddd", "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" ]
    transactions = []
    num_normal = 50; num_suspicious = 5; num_high_value = 3
    print(f"Generating {num_normal} normal transactions...")
    for i in range(num_normal):
        tx = { "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}", "from_address": random.choice(normal_wallets), "to_address": random.choice(normal_wallets), "value_eth": round(random.uniform(0.1, 20), 4), "gas_price": round(random.uniform(20, 100), 2), "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z" }
        if tx["from_address"] == tx["to_address"] and random.random() > 0.1: tx["to_address"] = random.choice([w for w in normal_wallets if w != tx["from_address"]])
        transactions.append(tx)
    print(f"Generating {num_suspicious} suspicious transactions...")
    for i in range(num_suspicious):
        from_is_malicious = random.random() > 0.5; to_is_malicious = random.random() > 0.5
        if not from_is_malicious and not to_is_malicious:
            if random.random() > 0.5: from_is_malicious = True
            else: to_is_malicious = True
        tx = { "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}", "from_address": random.choice(malicious_wallets) if from_is_malicious else random.choice(normal_wallets), "to_address": random.choice(malicious_wallets) if to_is_malicious else random.choice(normal_wallets), "value_eth": round(random.uniform(30, 150), 4), "gas_price": round(random.uniform(50, 150), 2), "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z" }
        if tx["from_address"] == tx["to_address"] and from_is_malicious: tx["to_address"] = random.choice(normal_wallets)
        transactions.append(tx)
    print(f"Generating {num_high_value} high-value transactions...")
    for i in range(num_high_value):
        tx = { "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}", "from_address": random.choice(normal_wallets), "to_address": random.choice(normal_wallets), "value_eth": round(random.uniform(60, 200), 4), "gas_price": round(random.uniform(20, 100), 2), "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z" }
        transactions.append(tx)
    random.shuffle(transactions)
    try:
        with open(filepath, 'w') as f: json.dump(transactions, f, indent=2)
        print(f"Created {len(transactions)} mock transactions. Saved to: {filepath}")
    except IOError as e: print(f"Error writing to {filepath}: {e}"); return None, None
    return malicious_wallets, normal_wallets

def create_dark_web_wallet_list(malicious_wallets):
    """Create dark web wallet threat intelligence file"""
    # ... (function remains the same) ...
    if malicious_wallets is None: print("Skipping dark web list creation due to previous error."); return
    print("\n=== Creating Dark Web Wallet List ===")
    filepath = 'data/dark_web_wallets.txt'
    try:
        with open(filepath, 'w') as f:
            for wallet in malicious_wallets: f.write(f"{wallet}\n")
        print(f"Created dark web wallet list with {len(malicious_wallets)} addresses. Saved to: {filepath}")
    except IOError as e: print(f"Error writing to {filepath}: {e}")

def create_wallet_profiles_db(normal_wallets):
    """Create SQLite database with wallet historical profiles"""
    # ... (function remains the same) ...
    if normal_wallets is None: print("Skipping wallet profile DB creation due to previous error."); return
    print("\n=== Creating Wallet Profiles Database ===")
    filepath = 'data/wallet_profiles.db'; conn = None
    try:
        conn = sqlite3.connect(filepath); cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS wallet_profiles")
        cursor.execute(""" CREATE TABLE wallet_profiles ( wallet_address TEXT PRIMARY KEY, avg_transaction_value REAL, transaction_count INTEGER, first_seen TEXT, last_updated TEXT ) """)
        print(f"Populating profiles for {len(normal_wallets)} wallets...")
        for wallet in normal_wallets:
            avg_value = round(random.uniform(1, 15), 2); tx_count = random.randint(10, 100)
            first_seen_date = f"2025-0{random.randint(1,9)}-{random.randint(1,28):02d}T00:00:00Z"
            cursor.execute(""" INSERT OR REPLACE INTO wallet_profiles (wallet_address, avg_transaction_value, transaction_count, first_seen, last_updated) VALUES (?, ?, ?, ?, ?) """, (wallet, avg_value, tx_count, first_seen_date, "2025-10-01T00:00:00Z"))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM wallet_profiles"); count = cursor.fetchone()[0]
        print(f"Created wallet_profiles table with {count} wallet histories. Saved to: {filepath}")
    except sqlite3.Error as e: print(f"SQLite error during DB creation: {e}")
    except Exception as e: print(f"Unexpected error during DB creation: {e}")
    finally:
        if conn: conn.close()

def train_behavior_model():
    """Train IsolationForest model on blockchain transaction features"""
    # ... (function remains the same) ...
    print("\n=== Training Behavior Model ===")
    transactions_filepath = 'data/mock_blockchain_transactions.json'; model_filepath = 'models/behavior_model.pkl'
    try:
        with open(transactions_filepath, 'r') as f: transactions = json.load(f)
    except FileNotFoundError: print(f"Error: {transactions_filepath} not found."); return
    except json.JSONDecodeError as e: print(f"Error: Invalid JSON in {transactions_filepath}: {e}"); return
    features = []
    for tx in transactions: features.append([ tx.get('value_eth', 0), tx.get('gas_price', 0) ])
    if not features: print("No features extracted. Skipping model training."); return
    features = np.array(features)
    print(f"Using {features.shape[0]} samples and {features.shape[1]} features for training.")
    model = IsolationForest(contamination='auto', random_state=42, n_estimators=100); model.fit(features)
    try:
        with open(model_filepath, 'wb') as f: pickle.dump(model, f)
        print(f"Trained IsolationForest model. Saved to: {model_filepath}")
    except IOError as e: print(f"Error saving model to {model_filepath}: {e}")

def main():
    """Main setup function"""
    start_time = datetime.now() # Now datetime is defined
    print("=" * 60)
    print(f"BLOCKCHAIN FRAUD DETECTION MVP - SETUP ({start_time.strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 60)

    # Create directories first
    create_directories()

    # Create all mock data and models
    malicious_wallets, normal_wallets = create_mock_transactions()
    create_dark_web_wallet_list(malicious_wallets)
    create_wallet_profiles_db(normal_wallets)
    train_behavior_model()

    end_time = datetime.now() # datetime is defined here too
    duration = end_time - start_time
    print("\n" + "=" * 60)
    print(f"SETUP COMPLETE! (Duration: {duration})")
    print("=" * 60)
    print("\nBackend setup is done. You can now start the backend and frontend.")
    print("1. Ensure frontend dependencies are installed (`npm install` in 'frontend').")
    print("2. Run `start_backend.bat` (or equivalent).")
    print("3. In a new terminal, run `start_frontend.bat` (or equivalent).")
    print("=" * 60)


if __name__ == "__main__":
    main()