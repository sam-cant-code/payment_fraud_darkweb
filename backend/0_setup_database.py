"""
Setup Script for Blockchain Fraud Detection MVP
Creates enhanced mock data, database, trains the ML model, and ADDS GROUND TRUTH LABELS
(Version with more data, realistic values, and diverse fraud types - CORRECTED)
"""

import json
import sqlite3
import pickle
import random
import os
from sklearn.ensemble import IsolationForest
import numpy as np
from datetime import datetime
import math # Needed for lognormvariate

# --- Define Wallet Sets Globally (or pass them around) ---
# Defining them here makes them accessible to multiple functions easily
MALICIOUS_WALLETS = {
    f"0xbad{i:038x}" for i in range(5) # 5 known bad actors
}
MIXER_ADDRESSES = {
    f"0xmix{i:038x}" for i in range(2) # 2 hypothetical mixer addresses
}
NORMAL_WALLETS = {
    f"0xnorm{i:037x}" for i in range(20) # 20 normal wallets
}
# Ensure no overlap
MALICIOUS_WALLETS -= NORMAL_WALLETS
MIXER_ADDRESSES -= NORMAL_WALLETS
MALICIOUS_WALLETS -= MIXER_ADDRESSES

ALL_NORMAL_LIST = list(NORMAL_WALLETS)
ALL_MALICIOUS_LIST = list(MALICIOUS_WALLETS)
ALL_MIXER_LIST = list(MIXER_ADDRESSES)
# Combine known malicious and mixers for the threat list
COMBINED_THREAT_LIST = list(MALICIOUS_WALLETS | MIXER_ADDRESSES)


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'output']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def create_mock_transactions():
    """Create enhanced mock blockchain transactions JSON file WITH is_fraud labels"""
    print("\n=== Creating Enhanced Mock Blockchain Transactions ===")
    filepath = 'data/mock_blockchain_transactions.json'

    # Use the globally defined wallet lists
    malicious_wallets = ALL_MALICIOUS_LIST
    mixer_addresses = ALL_MIXER_LIST
    normal_wallets = ALL_NORMAL_LIST

    transactions = []

    # --- Configuration for Generation ---
    num_normal = 500
    num_malicious_interaction = 15
    num_high_value = 5
    num_mixer_interaction = 10
    num_anomalous_gas = 10
    num_subtle_high_value = 10

    # --- Helper for Realistic Value Distribution (Log-Normal) ---
    log_mu = math.log(1.5)
    log_sigma = 1.5
    def get_realistic_value():
        val = random.lognormvariate(log_mu, log_sigma)
        return max(0.001, min(val, 500.0))

    # --- Generate Normal Transactions ---
    print(f"Generating {num_normal} normal transactions...")
    for i in range(num_normal):
        val = get_realistic_value()
        if val > 30: val = random.uniform(0.1, 30)
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(val, 6),
            "gas_price": round(random.uniform(15, 80), 2),
            "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z",
            "is_fraud": 0
        }
        if tx["from_address"] == tx["to_address"] and random.random() > 0.05:
            tx["to_address"] = random.choice([w for w in normal_wallets if w != tx["from_address"]])
        transactions.append(tx)

    # --- Generate Malicious Interaction Transactions ---
    print(f"Generating {num_malicious_interaction} transactions involving malicious wallets...")
    for i in range(num_malicious_interaction):
        use_malicious_sender = random.random() < 0.5
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(malicious_wallets) if use_malicious_sender else random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets) if use_malicious_sender else random.choice(malicious_wallets),
            "value_eth": round(get_realistic_value(), 6),
            "gas_price": round(random.uniform(20, 100), 2),
            "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z",
            "is_fraud": 1
        }
        transactions.append(tx)

    # --- Generate High Value Transactions (Fraud Simulation) ---
    print(f"Generating {num_high_value} suspicious transactions (very high value)...")
    for i in range(num_high_value):
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(random.uniform(101, 300), 4),
            "gas_price": round(random.uniform(20, 100), 2),
            "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z",
            "is_fraud": 1
        }
        transactions.append(tx)

    # --- Generate Mixer Interaction Transactions ---
    print(f"Generating {num_mixer_interaction} transactions involving mixers...")
    for i in range(num_mixer_interaction):
        use_mixer_as_recipient = random.random() < 0.7
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets) if use_mixer_as_recipient else random.choice(mixer_addresses),
            "to_address": random.choice(mixer_addresses) if use_mixer_as_recipient else random.choice(normal_wallets),
            "value_eth": round(get_realistic_value(), 6),
            "gas_price": round(random.uniform(30, 120), 2),
            "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z",
            "is_fraud": 1
        }
        transactions.append(tx)

    # --- Generate Anomalous High Gas Transactions ---
    print(f"Generating {num_anomalous_gas} transactions with anomalous gas prices...")
    for i in range(num_anomalous_gas):
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(random.uniform(0.5, 10), 4),
            "gas_price": round(random.uniform(200, 500), 2),
            "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z",
            "is_fraud": 1
        }
        transactions.append(tx)

    # --- Generate Subtle High Value Transactions ---
    print(f"Generating {num_subtle_high_value} transactions with subtly high values...")
    for i in range(num_subtle_high_value):
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(random.uniform(30, 60), 4),
            "gas_price": round(random.uniform(20, 100), 2),
            "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z",
            "is_fraud": 1
        }
        transactions.append(tx)

    # --- Finalize and Save ---
    random.shuffle(transactions)
    total_generated = len(transactions)
    total_fraud = sum(1 for tx in transactions if tx['is_fraud'] == 1)
    print(f"\nTotal transactions generated: {total_generated}")
    # --- FIX: Removed Chinese characters that caused UnicodeEncodeError ---
    print(f"    - Breakdown: Normal: {total_generated - total_fraud}, Fraudulent: {total_fraud}")
    # --- END FIX ---

    try:
        with open(filepath, 'w') as f:
            json.dump(transactions, f, indent=2)
        print(f"Saved enhanced mock transactions with labels to: {filepath}")
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")
        return [], []
    return list(MALICIOUS_WALLETS | MIXER_ADDRESSES), list(NORMAL_WALLETS)


def create_dark_web_wallet_list(threat_wallets): # Takes the combined list
    """Create dark web wallet threat intelligence file"""
    if not threat_wallets:
        print("No threat wallets provided. Skipping dark web list creation.")
        filepath = 'data/dark_web_wallets.txt'
        try:
            with open(filepath, 'w') as f: pass
            print(f"Created/Emptied dark web wallet list: {filepath}")
        except IOError as e: print(f"Error creating/emptying {filepath}: {e}")
        return

    print("\n=== Creating Dark Web Wallet List ===")
    filepath = 'data/dark_web_wallets.txt'
    try:
        # Use the combined list passed as argument
        with open(filepath, 'w') as f:
            for wallet in sorted(list(threat_wallets)): # Sort for consistency
                 f.write(f"{wallet.lower()}\n")
        print(f"Created dark web wallet list with {len(threat_wallets)} addresses. Saved to: {filepath}")
    except IOError as e: print(f"Error writing to {filepath}: {e}")

def create_wallet_profiles_db(normal_wallets):
    """Create SQLite database with wallet historical profiles"""
    if not normal_wallets:
        print("No normal wallets provided. Skipping wallet profile DB creation.")
        return
    print("\n=== Creating Wallet Profiles Database ===")
    filepath = 'data/wallet_profiles.db'; conn = None
    try:
        conn = sqlite3.connect(filepath); cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS wallet_profiles")
        cursor.execute(""" CREATE TABLE wallet_profiles (
                              wallet_address TEXT PRIMARY KEY,
                              avg_transaction_value REAL,
                              transaction_count INTEGER,
                              first_seen TEXT,
                              last_updated TEXT
                           ) """)
        print(f"Populating profiles for {len(normal_wallets)} wallets...")
        for wallet in normal_wallets:
            log_mu_prof = math.log(1.0)
            log_sigma_prof = 1.0
            avg_value = round(max(0.01, min(random.lognormvariate(log_mu_prof, log_sigma_prof), 50.0)), 4)
            tx_count = random.randint(5, 200)
            first_seen_date = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T00:00:00Z"
            cursor.execute(""" INSERT OR REPLACE INTO wallet_profiles
                               (wallet_address, avg_transaction_value, transaction_count, first_seen, last_updated)
                               VALUES (?, ?, ?, ?, ?) """,
                           (wallet.lower(), avg_value, tx_count, first_seen_date, "2025-09-01T00:00:00Z"))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM wallet_profiles"); count = cursor.fetchone()[0]
        print(f"Created wallet_profiles table with {count} wallet histories. Saved to: {filepath}")
    except sqlite3.Error as e: print(f"SQLite error during DB creation: {e}")
    except Exception as e: print(f"Unexpected error during DB creation: {e}")
    finally:
        if conn: conn.close()

def train_behavior_model():
    """Train IsolationForest model ONLY on NORMAL blockchain transaction features"""
    print("\n=== Training Behavior Model (on Normal Data Only) ===")
    transactions_filepath = 'data/mock_blockchain_transactions.json'
    model_filepath = 'models/behavior_model.pkl'
    try:
        with open(transactions_filepath, 'r') as f:
            transactions = json.load(f)
    except FileNotFoundError: print(f"Error: {transactions_filepath} not found."); return
    except json.JSONDecodeError as e: print(f"Error: Invalid JSON in {transactions_filepath}: {e}"); return
    except Exception as e: print(f"Error reading {transactions_filepath}: {e}"); return

    features = []
    for tx in transactions:
        if tx.get('is_fraud', 1) == 0:
            features.append([
                tx.get('value_eth', 0),
                tx.get('gas_price', 0)
            ])

    if not features:
        print("Error: No normal samples (is_fraud=0) found in mock data. Cannot train model.")
        if os.path.exists(model_filepath):
             try: os.remove(model_filepath); print(f"Removed potentially outdated model file: {model_filepath}")
             except OSError as e: print(f"Error removing old model file: {e}")
        return

    features = np.array(features)
    print(f"Training IsolationForest using {features.shape[0]} 'normal' samples and {features.shape[1]} features.")

    model = IsolationForest(contamination='auto', random_state=42, n_estimators=150)
    model.fit(features)

    try:
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Trained IsolationForest model on normal data. Saved to: {model_filepath}")
    except IOError as e: print(f"Error saving model to {model_filepath}: {e}")
    except Exception as e: print(f"Unexpected error saving model: {e}")


def main():
    """Main setup function"""
    start_time = datetime.now()
    print("=" * 60)
    print(f"BLOCKCHAIN FRAUD DETECTION MVP - ENHANCED SETUP ({start_time.strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 60)

    create_directories()

    # Create mock data returns (combined_threat_list, normal_list)
    combined_threat_list, normal_list = create_mock_transactions()

    # Pass the correct lists to subsequent functions
    train_behavior_model() # Uses the generated JSON file
    create_dark_web_wallet_list(combined_threat_list) # Use the combined list
    create_wallet_profiles_db(normal_list) # Use only the normal list

    end_time = datetime.now()
    duration = end_time - start_time
    print("\n" + "=" * 60)
    print(f"ENHANCED SETUP COMPLETE! (Duration: {duration})")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Ensure frontend dependencies are installed (`npm install` in 'frontend').")
    print("2. Run `start_backend.bat` (or `python backend/app.py`).")
    print("3. In a new terminal, run `start_frontend.bat` (or `npm run dev` in 'frontend').")
    print("4. To process mock data & check accuracy:")
    print("   a. Activate backend venv: `cd backend` then `.\\venv\\Scripts\\activate` (Win) or `source venv/bin/activate` (Mac/Linux)")
    print("   b. Run simulation on mock data: `python 1_run_simulation.py`")
    print("   c. Calculate accuracy: `python 3_calculate_accuracy.py`")
    print("=" * 60)

if __name__ == "__main__":
    main()
