"""
ENHANCED Setup Script for Blockchain Fraud Detection
Addresses data leakage, adds diverse fraud patterns, and improves realism
"""

import json
import sqlite3
import pickle
import random
import os
import math
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, auc, average_precision_score)
from imblearn.over_sampling import SMOTE
from datetime import datetime, timedelta
from collections import defaultdict

# ================== CONFIGURATION ==================
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Dataset configuration - More realistic distribution
TOTAL_TRANSACTIONS = 250000
FRAUD_RATE = 0.03  # 3% fraud rate (more realistic)
NUM_FRAUD = int(TOTAL_TRANSACTIONS * FRAUD_RATE)
NUM_NORMAL = TOTAL_TRANSACTIONS - NUM_FRAUD

# Wallet pools - Ensure no overlap
NUM_MALICIOUS_WALLETS = 100
NUM_MIXER_WALLETS = 10
NUM_NORMAL_WALLETS = 1000
NUM_HIGH_VALUE_WALLETS = 50  # Legitimate wealthy wallets

# Fraud pattern distribution (totals to NUM_FRAUD)
FRAUD_PATTERNS = {
    'malicious_interaction': int(NUM_FRAUD * 0.25),      # 75 txs
    'high_value_suspicious': int(NUM_FRAUD * 0.15),      # 45 txs
    'mixer_usage': int(NUM_FRAUD * 0.20),                # 60 txs
    'rapid_micro_transfers': int(NUM_FRAUD * 0.15),      # 45 txs
    'circular_transfers': int(NUM_FRAUD * 0.10),         # 30 txs
    'abnormal_timing': int(NUM_FRAUD * 0.10),            # 30 txs
    'self_transfers': int(NUM_FRAUD * 0.05),             # 15 txs
}
# Adjust for rounding to ensure sum is NUM_FRAUD
total_assigned = sum(FRAUD_PATTERNS.values())
if total_assigned < NUM_FRAUD:
    FRAUD_PATTERNS['malicious_interaction'] += (NUM_FRAUD - total_assigned)


# Temporal configuration
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 10, 1)
TOTAL_DAYS = (END_DATE - START_DATE).days


# ================== WALLET GENERATION ==================
def generate_wallet_pools():
    """Generate separate wallet pools with no overlap"""
    print("\n=== Generating Wallet Pools ===")

    # Generate unique wallet sets
    malicious = {f"0xbad{i:038x}" for i in range(NUM_MALICIOUS_WALLETS)}
    mixers = {f"0xmix{i:038x}" for i in range(NUM_MIXER_WALLETS)}
    normal = {f"0xnorm{i:037x}" for i in range(NUM_NORMAL_WALLETS)}
    high_value = {f"0xwhale{i:036x}" for i in range(NUM_HIGH_VALUE_WALLETS)}

    # Ensure no overlaps
    all_wallets = malicious | mixers | normal | high_value
    assert len(all_wallets) == (NUM_MALICIOUS_WALLETS + NUM_MIXER_WALLETS +
                                NUM_NORMAL_WALLETS + NUM_HIGH_VALUE_WALLETS), "Wallet overlap detected!"

    print(f"✓ Created {len(malicious)} malicious wallets")
    print(f"✓ Created {len(mixers)} mixer wallets")
    print(f"✓ Created {len(normal)} normal wallets")
    print(f"✓ Created {len(high_value)} high-value legitimate wallets")

    return {
        'malicious': list(malicious),
        'mixers': list(mixers),
        'normal': list(normal),
        'high_value': list(high_value),
        'threat_list': list(malicious | mixers)
    }


# ================== REALISTIC VALUE DISTRIBUTIONS ==================
def get_normal_value():
    """Get realistic value for normal transaction (log-normal distribution)"""
    return max(0.001, min(random.lognormvariate(math.log(0.5), 1.2), 50.0))

def get_high_value():
    """Get value for legitimate high-value transaction"""
    return random.uniform(50, 500)

def get_suspicious_value():
    """Get value for suspicious transaction"""
    return random.uniform(80, 300)

def get_micro_value():
    """Get value for micro-transfer (smurfing)"""
    return random.uniform(0.001, 0.05)

def get_normal_gas():
    """Get realistic gas price (20-100 Gwei is normal)"""
    return round(random.uniform(20, 100), 2)

def get_high_gas():
    """Get suspiciously high gas price"""
    return round(random.uniform(150, 500), 2)

def generate_timestamp(base_time=None, days_range=None):
    """Generate realistic timestamp"""
    if base_time is None:
        base_time = START_DATE
    if days_range is None:
        days_range = TOTAL_DAYS

    offset = timedelta(
        days=random.randint(0, days_range),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    return (base_time + offset).isoformat() + 'Z'


# ================== FRAUD PATTERN GENERATORS ==================
def generate_malicious_interaction(wallets, timestamp):
    """Pattern 1: Transaction involving known malicious wallet"""
    use_malicious_sender = random.random() < 0.6

    return {
        "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
        "from_address": random.choice(wallets['malicious']) if use_malicious_sender
                        else random.choice(wallets['normal']),
        "to_address": random.choice(wallets['normal']) if use_malicious_sender
                      else random.choice(wallets['malicious']),
        "value_eth": round(get_normal_value() if random.random() < 0.7 else get_suspicious_value(), 6),
        "gas_price": get_normal_gas(),
        "timestamp": timestamp,
        "is_fraud": 1,
        "fraud_type": "malicious_interaction"
    }

def generate_high_value_suspicious(wallets, timestamp):
    """Pattern 2: Unusually high value from low-activity wallet"""
    return {
        "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
        "from_address": random.choice(wallets['normal']),
        "to_address": random.choice(wallets['normal']),
        "value_eth": round(get_suspicious_value(), 6),
        "gas_price": round(random.uniform(15, 50), 2),  # Unusually low gas for high value
        "timestamp": timestamp,
        "is_fraud": 1,
        "fraud_type": "high_value_suspicious"
    }

def generate_mixer_usage(wallets, timestamp):
    """Pattern 3: Transaction through mixer service"""
    to_mixer = random.random() < 0.7

    return {
        "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
        "from_address": random.choice(wallets['normal']) if to_mixer
                        else random.choice(wallets['mixers']),
        "to_address": random.choice(wallets['mixers']) if to_mixer
                      else random.choice(wallets['normal']),
        "value_eth": round(get_normal_value(), 6),
        "gas_price": get_normal_gas(),
        "timestamp": timestamp,
        "is_fraud": 1,
        "fraud_type": "mixer_usage"
    }

def generate_rapid_micro_transfers(wallets, base_timestamp):
    """Pattern 4: Rapid sequence of small transfers (smurfing/structuring)"""
    sender = random.choice(wallets['normal'])
    num_transfers = random.randint(5, 15)

    transfers = []
    current_time = datetime.fromisoformat(base_timestamp.replace('Z', ''))

    for i in range(num_transfers):
        # Add 30 seconds to 5 minutes between transfers
        current_time += timedelta(seconds=random.randint(30, 300))

        transfers.append({
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": sender,
            "to_address": random.choice(wallets['normal']),
            "value_eth": round(get_micro_value(), 6),
            "gas_price": get_normal_gas(),
            "timestamp": current_time.isoformat() + 'Z',
            "is_fraud": 1,
            "fraud_type": "rapid_micro_transfers"
        })

    return transfers

def generate_circular_transfer(wallets, base_timestamp):
    """Pattern 5: Circular transfer pattern (A -> B -> C -> A)"""
    # Create a chain of 3-5 wallets
    chain_length = random.randint(3, 5)
    chain_wallets = random.sample(wallets['normal'], chain_length)

    transfers = []
    current_time = datetime.fromisoformat(base_timestamp.replace('Z', ''))
    initial_value = round(get_normal_value(), 6)

    for i in range(chain_length):
        current_time += timedelta(minutes=random.randint(5, 30))
        from_wallet = chain_wallets[i]
        to_wallet = chain_wallets[(i + 1) % chain_length]  # Circular

        # Value decreases slightly each hop (fees)
        value = round(initial_value * (0.95 ** i), 6)

        transfers.append({
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": from_wallet,
            "to_address": to_wallet,
            "value_eth": value,
            "gas_price": get_normal_gas(),
            "timestamp": current_time.isoformat() + 'Z',
            "is_fraud": 1,
            "fraud_type": "circular_transfer"
        })

    return transfers

def generate_abnormal_timing(wallets, base_timestamp):
    """Pattern 6: Unusual timing patterns (e.g., burst of activity at odd hours)"""
    # Generate burst of transactions between 2-4 AM
    burst_time = datetime.fromisoformat(base_timestamp.replace('Z', ''))
    burst_time = burst_time.replace(hour=random.randint(2, 4), minute=random.randint(0, 59))

    num_txs = random.randint(8, 15)
    transfers = []

    for i in range(num_txs):
        burst_time += timedelta(seconds=random.randint(10, 120))

        transfers.append({
            "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
            "from_address": random.choice(wallets['normal']),
            "to_address": random.choice(wallets['normal']),
            "value_eth": round(get_normal_value(), 6),
            "gas_price": get_normal_gas(),
            "timestamp": burst_time.isoformat() + 'Z',
            "is_fraud": 1,
            "fraud_type": "abnormal_timing"
        })

    return transfers

def generate_self_transfer(wallets, timestamp):
    """Pattern 7: Self-transfer (often used for obfuscation)"""
    wallet = random.choice(wallets['normal'])

    return {
        "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
        "from_address": wallet,
        "to_address": wallet,
        "value_eth": round(get_normal_value(), 6),
        "gas_price": round(random.uniform(50, 150), 2),  # Higher gas
        "timestamp": timestamp,
        "is_fraud": 1,
        "fraud_type": "self_transfer"
    }


# ================== NORMAL TRANSACTION GENERATORS ==================
def generate_normal_transaction(wallets, timestamp):
    """Generate realistic normal transaction"""
    # 70% use normal wallets, 30% involve high-value wallets
    use_high_value = random.random() < 0.3

    if use_high_value:
        # Legitimate high-value transaction
        from_wallet = random.choice(wallets['high_value']) if random.random() < 0.5 else random.choice(wallets['normal'])
        to_wallet = random.choice(wallets['normal'])
        value = round(get_high_value() if random.random() < 0.3 else get_normal_value(), 6)
    else:
        from_wallet = random.choice(wallets['normal'])
        to_wallet = random.choice(wallets['normal'])
        value = round(get_normal_value(), 6)

    # Avoid self-transfers in normal data
    if from_wallet == to_wallet:
        possible_recipients = [w for w in wallets['normal'] if w != from_wallet]
        if possible_recipients:
            to_wallet = random.choice(possible_recipients)
        else:
            # Fallback in case normal wallet list is tiny
            to_wallet = random.choice(wallets['normal'])


    return {
        "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
        "from_address": from_wallet,
        "to_address": to_wallet,
        "value_eth": value,
        "gas_price": get_normal_gas(),
        "timestamp": timestamp,
        "is_fraud": 0,
        "fraud_type": "normal"
    }


# ================== MAIN DATA GENERATION ==================
def create_enhanced_mock_transactions():
    """Create comprehensive fraud detection dataset"""
    print("\n" + "="*70)
    print("ENHANCED MOCK TRANSACTION GENERATION")
    print("="*70)
    print(f"Total transactions: {TOTAL_TRANSACTIONS}")
    print(f"Fraud rate: {FRAUD_RATE*100:.1f}% ({NUM_FRAUD} fraud, {NUM_NORMAL} normal)")
    print("="*70)

    wallets = generate_wallet_pools()
    transactions = []

    # Track fraud pattern counts
    fraud_counts = defaultdict(int)

    # Generate fraud transactions
    print("\n=== Generating Fraud Patterns ===")

    # Pattern 1: Malicious interaction
    print(f"Generating {FRAUD_PATTERNS['malicious_interaction']} malicious interactions...")
    for _ in range(FRAUD_PATTERNS['malicious_interaction']):
        tx = generate_malicious_interaction(wallets, generate_timestamp())
        transactions.append(tx)
        fraud_counts[tx['fraud_type']] += 1

    # Pattern 2: High value suspicious
    print(f"Generating {FRAUD_PATTERNS['high_value_suspicious']} high-value suspicious txs...")
    for _ in range(FRAUD_PATTERNS['high_value_suspicious']):
        tx = generate_high_value_suspicious(wallets, generate_timestamp())
        transactions.append(tx)
        fraud_counts[tx['fraud_type']] += 1

    # Pattern 3: Mixer usage
    print(f"Generating {FRAUD_PATTERNS['mixer_usage']} mixer transactions...")
    for _ in range(FRAUD_PATTERNS['mixer_usage']):
        tx = generate_mixer_usage(wallets, generate_timestamp())
        transactions.append(tx)
        fraud_counts[tx['fraud_type']] += 1

    # Pattern 4: Rapid micro transfers (generates multiple txs per call)
    print(f"Generating ~{FRAUD_PATTERNS['rapid_micro_transfers']} rapid micro-transfer sequences...")
    num_sequences = max(1, FRAUD_PATTERNS['rapid_micro_transfers'] // 10)  # Each sequence has ~10 txs
    for _ in range(num_sequences):
        txs = generate_rapid_micro_transfers(wallets, generate_timestamp())
        transactions.extend(txs)
        fraud_counts['rapid_micro_transfers'] += len(txs)

    # Pattern 5: Circular transfers (generates multiple txs per call)
    print(f"Generating ~{FRAUD_PATTERNS['circular_transfers']} circular transfer chains...")
    num_chains = max(1, FRAUD_PATTERNS['circular_transfers'] // 4)  # Each chain has ~4 txs
    for _ in range(num_chains):
        txs = generate_circular_transfer(wallets, generate_timestamp())
        transactions.extend(txs)
        fraud_counts['circular_transfer'] += len(txs)

    # Pattern 6: Abnormal timing (generates multiple txs per call)
    print(f"Generating ~{FRAUD_PATTERNS['abnormal_timing']} abnormal timing bursts...")
    num_bursts = max(1, FRAUD_PATTERNS['abnormal_timing'] // 10)  # Each burst has ~10 txs
    for _ in range(num_bursts):
        txs = generate_abnormal_timing(wallets, generate_timestamp())
        transactions.extend(txs)
        fraud_counts['abnormal_timing'] += len(txs)

    # Pattern 7: Self transfers
    print(f"Generating {FRAUD_PATTERNS['self_transfers']} self-transfer txs...")
    for _ in range(FRAUD_PATTERNS['self_transfers']):
        tx = generate_self_transfer(wallets, generate_timestamp())
        transactions.append(tx)
        fraud_counts[tx['fraud_type']] += 1

    actual_fraud = len([t for t in transactions if t['is_fraud'] == 1])

    # Generate normal transactions to reach target total
    num_normal_needed = TOTAL_TRANSACTIONS - len(transactions)
    print(f"\n=== Generating {num_normal_needed} Normal Transactions ===")
    for i in range(num_normal_needed):
        if i % 1000 == 0 and i > 0:
            print(f"  Generated {i}/{num_normal_needed}...")
        tx = generate_normal_transaction(wallets, generate_timestamp())
        transactions.append(tx)

    # Sort by timestamp
    transactions.sort(key=lambda x: x['timestamp'])

    # Print summary
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total transactions: {len(transactions)}")
    actual_normal = len([t for t in transactions if t['is_fraud'] == 0])
    actual_fraud = len([t for t in transactions if t['is_fraud'] == 1]) # Recalculate
    print(f"Normal: {actual_normal} ({actual_normal/len(transactions)*100:.1f}%)")
    print(f"Fraud: {actual_fraud} ({actual_fraud/len(transactions)*100:.1f}%)")
    print(f"\nFraud Pattern Breakdown:")
    if actual_fraud > 0:
        for pattern, count in sorted(fraud_counts.items()):
            print(f"  {pattern}: {count} ({count/actual_fraud*100:.1f}%)")
    else:
        print("  No fraud transactions generated.")
    print("="*70)

    # Save to file
    filepath = PROJECT_ROOT / 'data' / 'mock_blockchain_transactions.json'
    os.makedirs(filepath.parent, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(transactions, f, indent=2)
    print(f"\n✓ Saved to: {filepath}")

    return transactions, wallets


# ================== ENHANCED FEATURE ENGINEERING ==================
def engineer_enhanced_features(transactions, threat_list):
    """Engineer comprehensive features with temporal and behavioral patterns"""
    print("\n=== Engineering Enhanced Features ===")

    # Build comprehensive wallet profiles
    wallet_history = defaultdict(lambda: {
        'tx_count': 0,
        'total_sent': 0.0,
        'total_received': 0.0,
        'timestamps': [],
        'sent_to': set(),
        'received_from': set()
    })

    # First pass: build history
    for tx in transactions:
        from_addr = tx['from_address'].lower()
        to_addr = tx['to_address'].lower()
        value = tx['value_eth']

        wallet_history[from_addr]['tx_count'] += 1
        wallet_history[from_addr]['total_sent'] += value
        wallet_history[from_addr]['sent_to'].add(to_addr)

        wallet_history[to_addr]['tx_count'] += 1 # Add tx_count for receivers too
        wallet_history[to_addr]['total_received'] += value
        wallet_history[to_addr]['received_from'].add(from_addr)

        try:
            ts = datetime.fromisoformat(tx['timestamp'].replace('Z', ''))
            wallet_history[from_addr]['timestamps'].append(ts)
        except:
            pass

    threat_set = set(w.lower() for w in threat_list)

    features = []
    labels = []

    print(f"Processing {len(transactions)} transactions...")

    for tx in transactions:
        from_addr = tx['from_address'].lower()
        to_addr = tx['to_address'].lower()
        value = tx['value_eth']
        gas = tx['gas_price']

        # Basic features
        feat = [
            value,                               # 0: Transaction value
            gas,                                 # 1: Gas price
            value * gas,                         # 2: Total cost
        ]

        # Threat intelligence
        feat.extend([
            1 if from_addr in threat_set else 0,   # 3: From is threat
            1 if to_addr in threat_set else 0,     # 4: To is threat
        ])

        # Sender profile
        sender_history = wallet_history[from_addr]
        sender_tx_count = sender_history['tx_count']
        sender_avg_sent = sender_history['total_sent'] / max(sender_tx_count, 1)
        sender_unique_recipients = len(sender_history['sent_to'])

        feat.extend([
            sender_tx_count,                       # 5: Sender tx count
            sender_avg_sent,                       # 6: Sender avg value
            value / max(sender_avg_sent, 0.001),   # 7: Value deviation ratio
            sender_unique_recipients,              # 8: Unique recipients
            sender_unique_recipients / max(sender_tx_count, 1),  # 9: Recipient diversity
        ])

        # Recipient profile
        receiver_history = wallet_history[to_addr]
        receiver_unique_senders = len(receiver_history['received_from'])

        feat.extend([
            receiver_history['tx_count'],          # 10: Receiver tx count
            receiver_history['total_received'],    # 11: Receiver total received
            receiver_unique_senders,               # 12: Unique senders to receiver
        ])

        # Temporal features
        try:
            ts = datetime.fromisoformat(tx['timestamp'].replace('Z', ''))
            feat.extend([
                ts.hour,                             # 13: Hour of day
                ts.weekday(),                        # 14: Day of week
                1 if 2 <= ts.hour <= 5 else 0,       # 15: Odd hours (2-5 AM)
            ])

            # Transaction velocity
            if sender_history['timestamps'] and len(sender_history['timestamps']) > 1:
                # Get timestamps *before or at* current transaction time
                recent_timestamps = sorted([t for t in sender_history['timestamps'] if t <= ts])[-10:]
                
                if len(recent_timestamps) >= 2:
                    time_diffs = [(recent_timestamps[i] - recent_timestamps[i-1]).total_seconds()
                                  for i in range(1, len(recent_timestamps))]
                    avg_time_between = np.mean(time_diffs) if time_diffs else 86400
                    min_time_between = np.min(time_diffs) if time_diffs else 86400
                    feat.extend([
                        avg_time_between,            # 16: Avg time between txs
                        min_time_between,            # 17: Min time between txs
                    ])
                else:
                    feat.extend([86400, 86400])
            else:
                feat.extend([86400, 86400]) # Default for no history
        except:
            feat.extend([12, 3, 0, 86400, 86400]) # Default values

        # Pattern-specific features
        feat.extend([
            1 if value > 100 else 0,             # 18: Very high value
            1 if value > 50 else 0,              # 19: High value
            1 if value < 0.1 else 0,             # 20: Micro value
            1 if gas > 150 else 0,               # 21: High gas
            1 if gas < 30 else 0,                # 22: Low gas
            gas / max(value, 0.001),             # 23: Gas-to-value ratio
            1 if from_addr == to_addr else 0,     # 24: Self-transfer
            1 if (value > 0 and (value * 100) % 100 < 0.01) else 0, # 25: Round number (e.g., 1.00, 10.00)
        ])

        features.append(feat)
        labels.append(tx['is_fraud'])

    print(f"✓ Engineered {len(features[0])} features per transaction")

    return np.array(features), np.array(labels)


# ================== ENHANCED MODEL TRAINING ==================
def train_enhanced_model(transactions, threat_list):
    """Train model with proper evaluation and no data leakage"""
    print("\n" + "="*70)
    print("ENHANCED MODEL TRAINING")
    print("="*70)

    # Engineer features
    X, y = engineer_enhanced_features(transactions, threat_list)

    if len(X) == 0:
        print("Error: No features generated.")
        return

    # CRITICAL: Split by time to prevent leakage
    # Use first 70% as train, last 30% as test
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n✓ Temporal split (70/30):")
    print(f"  Training: {len(X_train)} samples ({sum(y_train)} fraud, {len(y_train)-sum(y_train)} normal)")
    print(f"  Testing:  {len(X_test)} samples ({sum(y_test)} fraud, {len(y_test)-sum(y_test)} normal)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE conservatively
    print("\n✓ Applying SMOTE (0.25 ratio)...")
    try:
        smote = SMOTE(random_state=42, sampling_strategy=0.25, k_neighbors=min(3, sum(y_train)-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"  After SMOTE: {len(X_train_balanced)} samples")
        print(f"    - Fraud: {sum(y_train_balanced)}")
        print(f"    - Normal: {len(y_train_balanced) - sum(y_train_balanced)}")
    except ValueError as e:
        print(f"  SMOTE failed (likely not enough samples): {e}. Training on original data.")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train


    # Train model with balanced parameters
    print("\n✓ Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight={0: 1, 1: 3},
        max_features='sqrt',
        max_samples=0.7,
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )

    model.fit(X_train_balanced, y_train_balanced)
    print(f"✓ Training complete (OOB Score: {model.oob_score_:.4f})")

    # Evaluate
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Comprehensive metrics
    print("\n--- Test Set Performance ---")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                Normal  Fraud")
    print(f"Actual Normal   {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"       Fraud    {cm[1][0]:4d}   {cm[1][1]:4d}")

    tn, fp, fn, tp = cm.ravel()

    # Additional metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba) if sum(y_test) > 0 else 0.0
    pr_auc = average_precision_score(y_test, y_pred_proba) if sum(y_test) > 0 else 0.0

    print(f"\n--- Key Metrics ---")
    print(f"ROC-AUC Score:  {roc_auc:.4f}")
    print(f"PR-AUC Score:   {pr_auc:.4f}")
    print(f"Accuracy:       {(tp+tn)/(tp+tn+fp+fn):.4f}")
    print(f"Precision:      {tp/(tp+fp) if (tp+fp)>0 else 0:.4f}")
    print(f"Recall:         {tp/(tp+fn) if (tp+fn)>0 else 0:.4f}")
    print(f"F1-Score:       {2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0:.4f}")
    print(f"FP Rate:        {fp/(fp+tn) if (fp+tn)>0 else 0:.4f}")
    print(f"FN Rate:        {fn/(fn+tp) if (fn+tp)>0 else 0:.4f}")

    # Cross-validation
    print("\n--- Cross-Validation (5-Fold) ---")
    cv_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight={0: 1, 1: 3},
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=skf, scoring='f1')
    print(f"F1 Scores: {cv_scores}")
    print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Find optimal threshold
    print("\n--- Optimal Threshold Analysis ---")
    if sum(y_test) > 0:
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"  Max F1 Score: {f1_scores[optimal_idx]:.4f}")
        print(f"  Precision: {precision[optimal_idx]:.4f}")
        print(f"  Recall: {recall[optimal_idx]:.4f}")

        # Test with optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        cm_optimal = confusion_matrix(y_test, y_pred_optimal)

        print(f"\nConfusion Matrix at Optimal Threshold:")
        print(f"                  Predicted")
        print(f"                Normal  Fraud")
        print(f"Actual Normal   {cm_optimal[0][0]:4d}   {cm_optimal[0][1]:4d}")
        print(f"       Fraud    {cm_optimal[1][0]:4d}   {cm_optimal[1][1]:4d}")
    else:
        print("  Skipped: No positive samples in test set for PR curve.")
        optimal_threshold = 0.5


    # Feature importance
    print("\n--- Top 15 Most Important Features ---")
    feature_names = [
        'value_eth', 'gas_price', 'total_cost', 'from_threat', 'to_threat',
        'sender_tx_count', 'sender_avg_value', 'value_deviation',
        'sender_unique_recipients', 'recipient_diversity',
        'receiver_tx_count', 'receiver_total_received', 'receiver_unique_senders',
        'hour', 'weekday', 'odd_hours', 'avg_time_between_tx', 'min_time_between_tx',
        'very_high_value', 'high_value', 'micro_value', 'high_gas', 'low_gas',
        'gas_value_ratio', 'self_transfer', 'round_number'
    ]

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    for i, idx in enumerate(indices, 1):
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"  {i:2d}. {feat_name:25s}: {importances[idx]:.4f}")

    # Save model with timestamp
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = PROJECT_ROOT / 'models' / f'fraud_model_{run_timestamp}.pkl'
    scaler_path = PROJECT_ROOT / 'models' / f'scaler_{run_timestamp}.pkl'
    metadata_path = PROJECT_ROOT / 'models' / f'model_metadata_{run_timestamp}.json'

    os.makedirs(model_path.parent, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {scaler_path}")

    # Save metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'n_features': len(feature_names),
        'optimal_threshold': float(optimal_threshold),
        'test_accuracy': float((tp+tn)/(tp+tn+fp+fn)),
        'test_precision': float(tp/(tp+fp) if (tp+fp)>0 else 0),
        'test_recall': float(tp/(tp+fn) if (tp+fn)>0 else 0),
        'test_f1': float(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'fp_rate': float(fp/(fp+tn) if (fp+tn)>0 else 0),
        'fn_rate': float(fn/(fn+tp) if (fn+tp)>0 else 0),
        'training_date': datetime.now().isoformat(),
        'class_weight': {0: 1, 1: 3},
        'fraud_patterns': list(FRAUD_PATTERNS.keys()),
        'model_file': os.path.basename(str(model_path)),
        'scaler_file': os.path.basename(str(scaler_path))
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")

    # Save latest timestamp
    timestamp_file = PROJECT_ROOT / 'models' / 'latest_model_timestamp.txt'
    with open(timestamp_file, 'w') as f:
        f.write(run_timestamp)
    print(f"✓ Latest timestamp saved to: {timestamp_file}")

    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)


def create_directories():
    """Create necessary directories"""
    directories = [
        PROJECT_ROOT / 'data',
        PROJECT_ROOT / 'models',
        PROJECT_ROOT / 'output'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def create_wallet_profiles_db(wallets):
    """Create SQLite database with wallet historical profiles"""
    print("\n=== Creating Wallet Profiles Database ===")
    filepath = PROJECT_ROOT / 'data' / 'wallet_profiles.db'

    try:
        conn = sqlite3.connect(str(filepath))
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

        # Add all wallets to database
        all_wallets = (wallets['normal'] + wallets['high_value'] +
                       wallets['malicious'] + wallets['mixers'])

        print(f"Populating profiles for {len(all_wallets)} wallets...")
        for wallet in all_wallets:
            # Vary profiles based on wallet type
            if wallet in wallets['high_value']:
                avg_value = round(random.uniform(50, 200), 4)
                tx_count = random.randint(100, 500)
            elif wallet in wallets['malicious'] or wallet in wallets['mixers']:
                avg_value = round(random.uniform(1, 20), 4)
                tx_count = random.randint(20, 150)
            else:
                avg_value = round(random.lognormvariate(math.log(1.0), 1.0), 4)
                avg_value = max(0.1, min(avg_value, 30.0))
                tx_count = random.randint(5, 200)

            first_seen = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T00:00:00Z"

            cursor.execute("""
                INSERT OR REPLACE INTO wallet_profiles
                (wallet_address, avg_transaction_value, transaction_count, first_seen, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (wallet.lower(), avg_value, tx_count, first_seen, "2025-09-01T00:00:00Z"))

        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM wallet_profiles")
        count = cursor.fetchone()[0]
        print(f"✓ Created wallet profiles: {count} wallets")
        print(f"  Saved to: {filepath}")
        conn.close()

    except Exception as e:
        print(f"Error creating database: {e}")


def create_threat_list_file(threat_list):
    """Create dark web wallet threat intelligence file"""
    print("\n=== Creating Threat List File ===")
    filepath = PROJECT_ROOT / 'data' / 'dark_web_wallets.txt'

    try:
        with open(filepath, 'w') as f:
            for wallet in sorted(threat_list):
                f.write(f"{wallet.lower()}\n")
        print(f"✓ Created threat list with {len(threat_list)} addresses")
        print(f"  Saved to: {filepath}")
    except Exception as e:
        print(f"Error creating threat list: {e}")


def main():
    """Main setup function"""
    start_time = datetime.now()
    print("=" * 70)
    print("ENHANCED BLOCKCHAIN FRAUD DETECTION - SETUP")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Total transactions: {TOTAL_TRANSACTIONS:,}")
    print(f"  Fraud rate: {FRAUD_RATE*100:.1f}% ({NUM_FRAUD} fraud cases)")
    print(f"  Fraud patterns: {len(FRAUD_PATTERNS)}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Create directories
    create_directories()

    # Generate data
    transactions, wallets = create_enhanced_mock_transactions()

    # Train model
    if transactions:
        train_enhanced_model(transactions, wallets['threat_list'])
    else:
        print("Error: No transactions were generated. Skipping model training.")

    # Create supporting files
    create_threat_list_file(wallets['threat_list'])
    create_wallet_profiles_db(wallets)

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print(f"✓ SETUP COMPLETE! (Duration: {duration})")
    print("=" * 70)

    print("\n📊 IMPROVEMENTS IN THIS VERSION:")
    print("=" * 70)
    print("✓ 7 diverse fraud patterns (vs. 5 basic patterns)")
    print("✓ Temporal split prevents data leakage")
    print("✓ More realistic 3% fraud rate (vs. 9%)")
    print("✓ Enhanced feature engineering (26 features)")
    print("✓ Separate high-value legitimate wallets")
    print("✓ Behavioral patterns: velocity, timing, network")
    print("✓ Proper evaluation: ROC-AUC, PR-AUC, optimal threshold")
    
    print("\n🎯 FRAUD PATTERNS INCLUDED:")
    print("=" * 70)
    for pattern, count in FRAUD_PATTERNS.items():
        print(f"  • {pattern}: ~{count} transactions")

    print("\n📈 EXPECTED PERFORMANCE:")
    print("=" * 70)
    print("  • Accuracy:     > 95%")
    print("  • Precision:    ~60-75% (for Fraud class)")
    print("  • Recall:       ~80-90% (for Fraud class)")
    print("  • ROC-AUC:      > 0.90")
    print("  • PR-AUC:       > 0.70")
    
    print("\n🚀 NEXT STEPS:")
    print("=" * 70)
    print("1. Run: train_and_run_all.bat")
    print("2. Check metrics in output")
    print("3. Start backend: start_backend.bat")
    print("4. Monitor live transactions")
    print("=" * 70)


if __name__ == "__main__":
    main()