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


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'output']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def create_mock_transactions():
    """Create mock blockchain transactions JSON file"""
    print("\n=== Creating Mock Blockchain Transactions ===")
    
    # Known malicious wallets (will be added to dark web list)
    malicious_wallets = [
        "0x1234567890abcdef1234567890abcdef12345678",
        "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "0x9876543210fedcba9876543210fedcba98765432"
    ]
    
    # Normal wallets
    normal_wallets = [
        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "0xcccccccccccccccccccccccccccccccccccccccc",
        "0xdddddddddddddddddddddddddddddddddddddddd",
        "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    ]
    
    transactions = []
    
    # Create 50 normal transactions
    for i in range(50):
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(random.uniform(0.1, 20), 4),
            "gas_price": round(random.uniform(20, 100), 2),
            "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z"
        }
        transactions.append(tx)
    
    # Create 5 suspicious transactions with dark web wallets
    print("Adding suspicious transactions with dark web wallets...")
    for i in range(5):
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(malicious_wallets) if random.random() > 0.5 else random.choice(normal_wallets),
            "to_address": random.choice(malicious_wallets) if random.random() > 0.5 else random.choice(normal_wallets),
            "value_eth": round(random.uniform(30, 150), 4),
            "gas_price": round(random.uniform(20, 100), 2),
            "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z"
        }
        transactions.append(tx)
    
    # Create 3 high-value transactions
    print("Adding high-value transactions...")
    for i in range(3):
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(random.uniform(60, 200), 4),
            "gas_price": round(random.uniform(20, 100), 2),
            "timestamp": f"2025-10-{random.randint(1, 18):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z"
        }
        transactions.append(tx)
    
    # Shuffle transactions
    random.shuffle(transactions)
    
    # Save to JSON file
    with open('data/mock_blockchain_transactions.json', 'w') as f:
        json.dump(transactions, f, indent=2)
    
    print(f"Created {len(transactions)} mock transactions")
    print(f"Saved to: data/mock_blockchain_transactions.json")
    
    return malicious_wallets, normal_wallets


def create_dark_web_wallet_list(malicious_wallets):
    """Create dark web wallet threat intelligence file"""
    print("\n=== Creating Dark Web Wallet List ===")
    
    with open('data/dark_web_wallets.txt', 'w') as f:
        for wallet in malicious_wallets:
            f.write(f"{wallet}\n")
    
    print(f"Created dark web wallet list with {len(malicious_wallets)} addresses")
    print(f"Saved to: data/dark_web_wallets.txt")


def create_wallet_profiles_db(normal_wallets):
    """Create SQLite database with wallet historical profiles"""
    print("\n=== Creating Wallet Profiles Database ===")
    
    # Connect to database (creates file if doesn't exist)
    conn = sqlite3.connect('data/wallet_profiles.db')
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wallet_profiles (
            wallet_address TEXT PRIMARY KEY,
            avg_transaction_value REAL,
            transaction_count INTEGER,
            last_updated TEXT
        )
    """)
    
    # Insert historical data for normal wallets
    for wallet in normal_wallets:
        avg_value = round(random.uniform(1, 15), 2)
        tx_count = random.randint(10, 100)
        
        # --- THIS IS THE CORRECTED LINE ---
        cursor.execute("""
            INSERT OR REPLACE INTO wallet_profiles 
            (wallet_address, avg_transaction_value, transaction_count, last_updated)
            VALUES (?, ?, ?, ?)
        """, (wallet, avg_value, tx_count, "2025-10-01T00:00:00Z"))
    
    conn.commit()
    
    # Verify data
    cursor.execute("SELECT COUNT(*) FROM wallet_profiles")
    count = cursor.fetchone()[0]
    
    print(f"Created wallet_profiles table with {count} wallet histories")
    print(f"Saved to: data/wallet_profiles.db")
    
    conn.close()


def train_behavior_model():
    """Train IsolationForest model on blockchain transaction features"""
    print("\n=== Training Behavior Model ===")
    
    # Load the mock transactions
    try:
        with open('data/mock_blockchain_transactions.json', 'r') as f:
            transactions = json.load(f)
    except FileNotFoundError:
        print("Error: mock_blockchain_transactions.json not found.")
        print("Please ensure setup is run in the correct directory.")
        return

    # Extract features for training
    features = []
    for tx in transactions:
        features.append([
            tx['value_eth'],
            tx['gas_price']
        ])
    
    if not features:
        print("No features found to train model. Skipping.")
        return

    features = np.array(features)
    
    # Train IsolationForest model
    # contamination is the expected proportion of outliers
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features)
    
    # Save model
    with open('models/behavior_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Trained IsolationForest model on {len(features)} transactions")
    print(f"Features used: value_eth, gas_price")
    print(f"Saved to: models/behavior_model.pkl")


def main():
    """Main setup function"""
    print("=" * 60)
    print("BLOCKCHAIN FRAUD DETECTION MVP - SETUP")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Create all mock data and models
    malicious_wallets, normal_wallets = create_mock_transactions()
    create_dark_web_wallet_list(malicious_wallets)
    create_wallet_profiles_db(normal_wallets)
    train_behavior_model()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python 1_run_simulation.py")
    print("2. Then: streamlit run 2_dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()