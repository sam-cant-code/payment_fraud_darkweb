"""
Setup Script for Blockchain Fraud Detection MVP
Creates enhanced mock data, database, trains the ML model, and ADDS GROUND TRUTH LABELS
(Version with 5000 transactions total - optimized distribution)
"""

import json
import sqlite3
import pickle
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np
from datetime import datetime, timedelta
import math
from collections import defaultdict

# --- Define Wallet Sets Globally ---
MALICIOUS_WALLETS = {f"0xbad{i:038x}" for i in range(10)}  # 10 bad actors
MIXER_ADDRESSES = {f"0xmix{i:038x}" for i in range(3)}     # 3 mixers
NORMAL_WALLETS = {f"0xnorm{i:037x}" for i in range(50)}    # 50 normal wallets

MALICIOUS_WALLETS -= NORMAL_WALLETS
MIXER_ADDRESSES -= NORMAL_WALLETS
MALICIOUS_WALLETS -= MIXER_ADDRESSES

ALL_NORMAL_LIST = list(NORMAL_WALLETS)
ALL_MALICIOUS_LIST = list(MALICIOUS_WALLETS)
ALL_MIXER_LIST = list(MIXER_ADDRESSES)
COMBINED_THREAT_LIST = list(MALICIOUS_WALLETS | MIXER_ADDRESSES)


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['../data', '../models', '../output']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def create_mock_transactions():
    """Create enhanced mock blockchain transactions JSON file WITH is_fraud labels"""
    print("\n=== Creating Mock Blockchain Transactions (5000 total) ===")
    filepath = '../data/mock_blockchain_transactions.json'

    malicious_wallets = ALL_MALICIOUS_LIST
    mixer_addresses = ALL_MIXER_LIST
    normal_wallets = ALL_NORMAL_LIST

    transactions = []

    # --- Configuration for 5000 Total Transactions ---
    num_normal = 4550                   # Normal transactions (91%)
    num_malicious_interaction = 150     # Transactions with bad actors
    num_high_value = 50                 # Very high value (>100 ETH)
    num_mixer_interaction = 100         # Mixer transactions
    num_anomalous_gas = 100             # Unusual gas prices
    num_subtle_high_value = 50          # Moderately high value (30-60 ETH)
    # Total: 5000 transactions (450 fraud = 9%)

    # Helper for Realistic Value Distribution
    log_mu = math.log(1.5)
    log_sigma = 1.5
    
    def get_realistic_value():
        val = random.lognormvariate(log_mu, log_sigma)
        return max(0.001, min(val, 500.0))

    base_time = datetime(2025, 10, 1)

    # Generate Normal Transactions
    print(f"Generating {num_normal} normal transactions...")
    for i in range(num_normal):
        val = get_realistic_value()
        if val > 30:
            val = random.uniform(0.1, 30)
        
        time_offset = timedelta(
            days=random.randint(0, 17),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        timestamp = (base_time + time_offset).isoformat() + 'Z'
        
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(val, 6),
            "gas_price": round(random.uniform(15, 80), 2),
            "timestamp": timestamp,
            "is_fraud": 0
        }
        if tx["from_address"] == tx["to_address"]:
            tx["to_address"] = random.choice([w for w in normal_wallets if w != tx["from_address"]])
        transactions.append(tx)

    # Generate Malicious Interaction Transactions
    print(f"Generating {num_malicious_interaction} transactions involving malicious wallets...")
    for i in range(num_malicious_interaction):
        use_malicious_sender = random.random() < 0.5
        time_offset = timedelta(
            days=random.randint(0, 17),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        timestamp = (base_time + time_offset).isoformat() + 'Z'
        
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(malicious_wallets) if use_malicious_sender else random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets) if use_malicious_sender else random.choice(malicious_wallets),
            "value_eth": round(get_realistic_value(), 6),
            "gas_price": round(random.uniform(20, 100), 2),
            "timestamp": timestamp,
            "is_fraud": 1
        }
        transactions.append(tx)

    # Generate High Value Transactions (Fraud)
    print(f"Generating {num_high_value} suspicious transactions (very high value)...")
    for i in range(num_high_value):
        time_offset = timedelta(days=random.randint(0, 17), hours=random.randint(0, 23))
        timestamp = (base_time + time_offset).isoformat() + 'Z'
        
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(random.uniform(101, 300), 4),
            "gas_price": round(random.uniform(20, 100), 2),
            "timestamp": timestamp,
            "is_fraud": 1
        }
        transactions.append(tx)

    # Generate Mixer Interaction Transactions
    print(f"Generating {num_mixer_interaction} transactions involving mixers...")
    for i in range(num_mixer_interaction):
        use_mixer_as_recipient = random.random() < 0.7
        time_offset = timedelta(days=random.randint(0, 17), hours=random.randint(0, 23))
        timestamp = (base_time + time_offset).isoformat() + 'Z'
        
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets) if use_mixer_as_recipient else random.choice(mixer_addresses),
            "to_address": random.choice(mixer_addresses) if use_mixer_as_recipient else random.choice(normal_wallets),
            "value_eth": round(get_realistic_value(), 6),
            "gas_price": round(random.uniform(30, 120), 2),
            "timestamp": timestamp,
            "is_fraud": 1
        }
        transactions.append(tx)

    # Generate Anomalous High Gas Transactions
    print(f"Generating {num_anomalous_gas} transactions with anomalous gas prices...")
    for i in range(num_anomalous_gas):
        time_offset = timedelta(days=random.randint(0, 17), hours=random.randint(0, 23))
        timestamp = (base_time + time_offset).isoformat() + 'Z'
        
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(random.uniform(0.5, 10), 4),
            "gas_price": round(random.uniform(200, 500), 2),
            "timestamp": timestamp,
            "is_fraud": 1
        }
        transactions.append(tx)

    # Generate Subtle High Value Transactions
    print(f"Generating {num_subtle_high_value} transactions with subtly high values...")
    for i in range(num_subtle_high_value):
        time_offset = timedelta(days=random.randint(0, 17), hours=random.randint(0, 23))
        timestamp = (base_time + time_offset).isoformat() + 'Z'
        
        tx = {
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(normal_wallets),
            "to_address": random.choice(normal_wallets),
            "value_eth": round(random.uniform(30, 60), 4),
            "gas_price": round(random.uniform(20, 100), 2),
            "timestamp": timestamp,
            "is_fraud": 1
        }
        transactions.append(tx)

    # Sort by timestamp
    transactions.sort(key=lambda x: x['timestamp'])
    
    total_generated = len(transactions)
    total_fraud = sum(1 for tx in transactions if tx['is_fraud'] == 1)
    print(f"\n✓ Total transactions generated: {total_generated}")
    print(f"  - Normal: {total_generated - total_fraud} ({(total_generated-total_fraud)/total_generated*100:.1f}%)")
    print(f"  - Fraudulent: {total_fraud} ({total_fraud/total_generated*100:.1f}%)")

    try:
        with open(filepath, 'w') as f:
            json.dump(transactions, f, indent=2)
        print(f"✓ Saved to: {filepath}")
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")
        return [], []
    
    return list(MALICIOUS_WALLETS | MIXER_ADDRESSES), list(NORMAL_WALLETS)


def engineer_features(transactions):
    """Engineer comprehensive features from raw transaction data"""
    print("\n=== Engineering Features ===")
    
    wallet_tx_count = defaultdict(int)
    wallet_total_value = defaultdict(float)
    wallet_timestamps = defaultdict(list)
    
    for tx in transactions:
        from_addr = tx['from_address'].lower()
        to_addr = tx['to_address'].lower()
        
        wallet_tx_count[from_addr] += 1
        wallet_total_value[from_addr] += tx['value_eth']
        
        try:
            ts = datetime.fromisoformat(tx['timestamp'].replace('Z', ''))
            wallet_timestamps[from_addr].append(ts)
        except:
            pass
    
    threat_set = set(w.lower() for w in COMBINED_THREAT_LIST)
    
    features = []
    labels = []
    
    print(f"Processing {len(transactions)} transactions...")
    
    for tx in transactions:
        from_addr = tx['from_address'].lower()
        to_addr = tx['to_address'].lower()
        value = tx['value_eth']
        gas = tx['gas_price']
        
        feat = [
            value,
            gas,
            value * gas,
        ]
        
        feat.append(1 if from_addr in threat_set else 0)
        feat.append(1 if to_addr in threat_set else 0)
        
        sender_tx_count = wallet_tx_count.get(from_addr, 0)
        sender_avg_value = wallet_total_value.get(from_addr, 0) / max(sender_tx_count, 1)
        
        feat.append(sender_tx_count)
        feat.append(sender_avg_value)
        feat.append(value / max(sender_avg_value, 0.001))
        
        try:
            ts = datetime.fromisoformat(tx['timestamp'].replace('Z', ''))
            feat.append(ts.hour)
            feat.append(ts.weekday())
            
            if from_addr in wallet_timestamps and len(wallet_timestamps[from_addr]) > 1:
                recent_timestamps = sorted(wallet_timestamps[from_addr])[-5:]
                if len(recent_timestamps) >= 2:
                    time_diffs = [(recent_timestamps[i] - recent_timestamps[i-1]).total_seconds() 
                                  for i in range(1, len(recent_timestamps))]
                    avg_time_between = np.mean(time_diffs)
                    feat.append(avg_time_between)
                else:
                    feat.append(86400)
            else:
                feat.append(86400)
        except:
            feat.append(12)
            feat.append(3)
            feat.append(86400)
        
        feat.append(1 if value > 100 else 0)
        feat.append(1 if value > 50 else 0)
        feat.append(1 if gas > 150 else 0)
        feat.append(gas / max(value, 0.001))
        
        features.append(feat)
        labels.append(tx['is_fraud'])
    
    print(f"✓ Engineered {len(features[0])} features per transaction")
    return np.array(features), np.array(labels)


def train_fraud_model():
    """Train a SUPERVISED fraud detection model with improved parameters"""
    print("\n=== Training Fraud Detection Model (Supervised Learning) ===")
    
    # --- FIX 1: Correct file paths to point to the parent data/ and models/ directories ---
    transactions_filepath = '../data/mock_blockchain_transactions.json'
    model_filepath = '../models/fraud_model.pkl'
    scaler_filepath = '../models/scaler.pkl'
    
    
    try:
        with open(transactions_filepath, 'r') as f:
            transactions = json.load(f)
        print(f"✓ Loaded transactions from: {transactions_filepath}")
    except FileNotFoundError:
        print(f"Error: {transactions_filepath} not found.")
        print("Please ensure mock data was created successfully.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return
    
    X, y = engineer_features(transactions)
    
    if len(X) == 0:
        print("Error: No features generated.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Dataset split:")
    print(f"  Training: {len(X_train)} samples ({sum(y_train)} fraud)")
    print(f"  Testing:  {len(X_test)} samples ({sum(y_test)} fraud)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✓ Applying SMOTE with conservative ratio...")
    # Use a less aggressive SMOTE ratio
    smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Only oversample to 50% of majority
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"  After SMOTE: {len(X_train_balanced)} samples")
    
    print("\n✓ Training Random Forest with better recall focus...")
    model = RandomForestClassifier(
        n_estimators=300,  # Increased from 200
        max_depth=20,  # Increased from 15 for more complexity
        min_samples_split=3,  # Reduced from 5 for finer splits
        min_samples_leaf=1,  # Reduced from 2
        class_weight={0: 1, 1: 3},  # Give more weight to fraud class
        random_state=42,
        n_jobs=-1,
        max_features='sqrt'  # Add this for better generalization
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Adjust decision threshold for better recall
    # Instead of 0.5, use 0.3 as threshold
    y_pred_adjusted = (y_pred_proba >= 0.3).astype(int)
    
    print("\nClassification Report (Standard Threshold 0.5):")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    print("\nClassification Report (Adjusted Threshold 0.3 for better recall):")
    print(classification_report(y_test, y_pred_adjusted, target_names=['Normal', 'Fraud']))
    
    print("\nConfusion Matrix (Adjusted Threshold):")
    cm = confusion_matrix(y_test, y_pred_adjusted)
    print(f"                Predicted")
    print(f"              Normal  Fraud")
    print(f"Actual Normal   {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"       Fraud    {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
    except:
        print("\nROC-AUC Score: Could not calculate")
    
    print("\nTop 10 Most Important Features:")
    feature_names = [
        'value_eth', 'gas_price', 'total_cost', 'from_threat', 'to_threat',
        'sender_tx_count', 'sender_avg_value', 'value_deviation', 
        'hour', 'weekday', 'tx_velocity', 'very_high_value', 
        'high_value', 'high_gas', 'gas_value_ratio'
    ]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"  {i}. {feat_name}: {importances[idx]:.4f}")
    
    try:
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # --- FIX 2: Correct the variable name from scaler_path to scaler_filepath ---
        with open(scaler_filepath, 'wb') as f:
            pickle.dump(scaler, f)
            
        print(f"\n✓ Model saved to: {model_filepath}")
        print(f"✓ Scaler saved to: {scaler_filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")


def create_dark_web_wallet_list(threat_wallets):
    """Create dark web wallet threat intelligence file"""
    if not threat_wallets:
        print("No threat wallets provided.")
        return

    print("\n=== Creating Dark Web Wallet List ===")
    filepath = '../data/dark_web_wallets.txt'
    try:
        with open(filepath, 'w') as f:
            for wallet in sorted(list(threat_wallets)):
                f.write(f"{wallet.lower()}\n")
        print(f"✓ Created threat list with {len(threat_wallets)} addresses")
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")


def create_wallet_profiles_db(normal_wallets):
    """Create SQLite database with wallet historical profiles"""
    if not normal_wallets:
        print("No normal wallets provided.")
        return
    
    print("\n=== Creating Wallet Profiles Database ===")
    filepath = '../data/wallet_profiles.db'
    conn = None
    
    try:
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS wallet_profiles")
        cursor.execute("""
            CREATE TABLE wallet_profiles (
                wallet_address TEXT PRIMARY KEY,
                avg_transaction_value REAL,
                transaction_count INTEGER,
                first_seen TEXT,
                last_updated TEXT
            )
        """)
        
        print(f"Populating profiles for {len(normal_wallets)} wallets...")
        for wallet in normal_wallets:
            log_mu_prof = math.log(1.0)
            log_sigma_prof = 1.0
            avg_value = round(max(0.01, min(random.lognormvariate(log_mu_prof, log_sigma_prof), 50.0)), 4)
            tx_count = random.randint(5, 200)
            first_seen_date = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T00:00:00Z"
            
            cursor.execute("""
                INSERT OR REPLACE INTO wallet_profiles
                (wallet_address, avg_transaction_value, transaction_count, first_seen, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (wallet.lower(), avg_value, tx_count, first_seen_date, "2025-09-01T00:00:00Z"))
        
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM wallet_profiles")
        count = cursor.fetchone()[0]
        print(f"✓ Created wallet profiles: {count} wallets")
    
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if conn:
            conn.close()


def main():
    """Main setup function"""
    start_time = datetime.now()
    print("=" * 60)
    print(f"BLOCKCHAIN FRAUD DETECTION - SETUP")
    print(f"Target: 5000 transactions")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    create_directories()
    combined_threat_list, normal_list = create_mock_transactions()
    train_fraud_model()
    create_dark_web_wallet_list(combined_threat_list)
    create_wallet_profiles_db(normal_list)

    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"✓ SETUP COMPLETE! (Duration: {duration})")
    print("=" * 60)
    print("\nCreated files:")
    print("  - ../data/mock_blockchain_transactions.json (5000 transactions)")
    print("  - ../models/fraud_model.pkl (Trained RandomForest)")
    print("  - ../models/scaler.pkl (Feature scaler)")
    print("  - ../data/dark_web_wallets.txt (Threat intelligence)")
    print("  - ../data/wallet_profiles.db (Historical profiles)")
    print("\nNext steps:")
    print("  1. Run 'start_backend.bat' to start the API server")
    print("  2. Run 'run_accuracy_check.bat' to test the model")
    print("  3. Start your React frontend")
    print("=" * 60)


if __name__ == "__main__":
    main()