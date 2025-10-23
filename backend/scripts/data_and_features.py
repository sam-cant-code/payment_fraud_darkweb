"""
Utility Script for Data Generation & Feature Engineering
This file is not run directly.
It acts as a shared library for:
- initialize_and_train.py
- evaluate_model.py
"""

import json
import sqlite3
import random
import os
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ================== CONFIGURATION ==================
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Dataset configuration
TOTAL_TRANSACTIONS = 250000
FRAUD_RATE = 0.03
NUM_FRAUD = int(TOTAL_TRANSACTIONS * FRAUD_RATE)
NUM_NORMAL = TOTAL_TRANSACTIONS - NUM_FRAUD

# Wallet pools
NUM_MALICIOUS_WALLETS = 100
NUM_MIXER_WALLETS = 10
NUM_NORMAL_WALLETS = 1000
NUM_HIGH_VALUE_WALLETS = 50

# Fraud pattern distribution
FRAUD_PATTERNS = {
    'malicious_interaction': int(NUM_FRAUD * 0.25),
    'high_value_suspicious': int(NUM_FRAUD * 0.15),
    'mixer_usage': int(NUM_FRAUD * 0.20),
    'rapid_micro_transfers': int(NUM_FRAUD * 0.15),
    'circular_transfers': int(NUM_FRAUD * 0.10),
    'abnormal_timing': int(NUM_FRAUD * 0.10),
    'self_transfers': int(NUM_FRAUD * 0.05),
}
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

    malicious = {f"0xbad{i:038x}" for i in range(NUM_MALICIOUS_WALLETS)}
    mixers = {f"0xmix{i:038x}" for i in range(NUM_MIXER_WALLETS)}
    normal = {f"0xnorm{i:037x}" for i in range(NUM_NORMAL_WALLETS)}
    high_value = {f"0xwhale{i:036x}" for i in range(NUM_HIGH_VALUE_WALLETS)}

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
        'threat_list': list(malicious | mixers),
        'all_wallets': list(all_wallets)
    }


# ================== REALISTIC VALUE DISTRIBUTIONS ==================
def get_normal_value():
    return max(0.001, min(random.lognormvariate(math.log(0.5), 1.2), 50.0))
def get_high_value():
    return random.uniform(50, 500)
def get_suspicious_value():
    return random.uniform(80, 300)
def get_micro_value():
    return random.uniform(0.001, 0.05)
def get_normal_gas():
    return round(random.uniform(20, 100), 2)
def get_high_gas():
    return round(random.uniform(150, 500), 2)
def generate_timestamp(base_time=None, days_range=None):
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
    return {
        "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
        "from_address": random.choice(wallets['normal']),
        "to_address": random.choice(wallets['normal']),
        "value_eth": round(get_suspicious_value(), 6),
        "gas_price": round(random.uniform(15, 50), 2),
        "timestamp": timestamp,
        "is_fraud": 1,
        "fraud_type": "high_value_suspicious"
    }
def generate_mixer_usage(wallets, timestamp):
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
    sender = random.choice(wallets['normal'])
    num_transfers = random.randint(5, 15)
    transfers = []
    current_time = datetime.fromisoformat(base_timestamp.replace('Z', ''))
    for i in range(num_transfers):
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
    chain_length = random.randint(3, 5)
    chain_wallets = random.sample(wallets['normal'], chain_length)
    transfers = []
    current_time = datetime.fromisoformat(base_timestamp.replace('Z', ''))
    initial_value = round(get_normal_value(), 6)
    for i in range(chain_length):
        current_time += timedelta(minutes=random.randint(5, 30))
        from_wallet = chain_wallets[i]
        to_wallet = chain_wallets[(i + 1) % chain_length]
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
    wallet = random.choice(wallets['normal'])
    return {
        "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
        "from_address": wallet,
        "to_address": wallet,
        "value_eth": round(get_normal_value(), 6),
        "gas_price": round(random.uniform(50, 150), 2),
        "timestamp": timestamp,
        "is_fraud": 1,
        "fraud_type": "self_transfer"
    }


# ================== NORMAL TRANSACTION GENERATORS ==================
def generate_normal_transaction(wallets, timestamp):
    use_high_value = random.random() < 0.3
    if use_high_value:
        from_wallet = random.choice(wallets['high_value']) if random.random() < 0.5 else random.choice(wallets['normal'])
        to_wallet = random.choice(wallets['normal'])
        value = round(get_high_value() if random.random() < 0.3 else get_normal_value(), 6)
    else:
        from_wallet = random.choice(wallets['normal'])
        to_wallet = random.choice(wallets['normal'])
        value = round(get_normal_value(), 6)
    while from_wallet == to_wallet:
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
def create_enhanced_mock_transactions(load_if_exists=False):
    """
    Create comprehensive fraud detection dataset.
    If load_if_exists=True, it will load from file instead of regenerating.
    """
    print("\n" + "="*70)
    print("ENHANCED MOCK TRANSACTION GENERATION")
    print("="*70)
    
    filepath = PROJECT_ROOT / 'data' / 'mock_blockchain_transactions.json'
    wallets = generate_wallet_pools() # Always need wallet lists

    if load_if_exists and filepath.exists():
        try:
            print(f"Loading existing dataset from {filepath}...")
            with open(filepath, 'r') as f:
                transactions = json.load(f)
            print(f"✓ Loaded {len(transactions)} transactions.")
            return transactions, wallets
        except Exception as e:
            print(f"Error loading {filepath}: {e}. Regenerating...")

    print(f"Generating {TOTAL_TRANSACTIONS} new transactions...")
    print(f"Fraud rate: {FRAUD_RATE*100:.1f}% ({NUM_FRAUD} fraud, {NUM_NORMAL} normal)")
    print("="*70)

    transactions = []
    fraud_counts = defaultdict(int)

    # ... (Fraud pattern generation logic) ...
    print("\n=== Generating Fraud Patterns ===")
    print(f"Generating {FRAUD_PATTERNS['malicious_interaction']} malicious interactions...")
    for _ in range(FRAUD_PATTERNS['malicious_interaction']):
        tx = generate_malicious_interaction(wallets, generate_timestamp())
        transactions.append(tx); fraud_counts[tx['fraud_type']] += 1
    print(f"Generating {FRAUD_PATTERNS['high_value_suspicious']} high-value suspicious txs...")
    for _ in range(FRAUD_PATTERNS['high_value_suspicious']):
        tx = generate_high_value_suspicious(wallets, generate_timestamp())
        transactions.append(tx); fraud_counts[tx['fraud_type']] += 1
    print(f"Generating {FRAUD_PATTERNS['mixer_usage']} mixer transactions...")
    for _ in range(FRAUD_PATTERNS['mixer_usage']):
        tx = generate_mixer_usage(wallets, generate_timestamp())
        transactions.append(tx); fraud_counts[tx['fraud_type']] += 1
    print(f"Generating ~{FRAUD_PATTERNS['rapid_micro_transfers']} rapid micro-transfer sequences...")
    num_sequences = max(1, FRAUD_PATTERNS['rapid_micro_transfers'] // 10)
    for _ in range(num_sequences):
        txs = generate_rapid_micro_transfers(wallets, generate_timestamp())
        transactions.extend(txs); fraud_counts['rapid_micro_transfers'] += len(txs)
    print(f"Generating ~{FRAUD_PATTERNS['circular_transfers']} circular transfer chains...")
    num_chains = max(1, FRAUD_PATTERNS['circular_transfers'] // 4)
    for _ in range(num_chains):
        txs = generate_circular_transfer(wallets, generate_timestamp())
        transactions.extend(txs); fraud_counts['circular_transfer'] += len(txs)
    print(f"Generating ~{FRAUD_PATTERNS['abnormal_timing']} abnormal timing bursts...")
    num_bursts = max(1, FRAUD_PATTERNS['abnormal_timing'] // 10)
    for _ in range(num_bursts):
        txs = generate_abnormal_timing(wallets, generate_timestamp())
        transactions.extend(txs); fraud_counts['abnormal_timing'] += len(txs)
    print(f"Generating {FRAUD_PATTERNS['self_transfers']} self-transfer txs...")
    for _ in range(FRAUD_PATTERNS['self_transfers']):
        tx = generate_self_transfer(wallets, generate_timestamp())
        transactions.append(tx); fraud_counts[tx['fraud_type']] += 1

    # Generate normal transactions
    num_normal_needed = TOTAL_TRANSACTIONS - len(transactions)
    print(f"\n=== Generating {num_normal_needed} Normal Transactions ===")
    for i in range(num_normal_needed):
        if i % 10000 == 0 and i > 0:
            print(f"  Generated {i}/{num_normal_needed}...")
        tx = generate_normal_transaction(wallets, generate_timestamp())
        transactions.append(tx)

    print("\nSorting transactions by timestamp...")
    transactions.sort(key=lambda x: x['timestamp'])

    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    actual_normal = len([t for t in transactions if t['is_fraud'] == 0])
    actual_fraud = len([t for t in transactions if t['is_fraud'] == 1])
    print(f"Total: {len(transactions)} (Normal: {actual_normal}, Fraud: {actual_fraud})")
    print(f"\nFraud Pattern Breakdown:")
    if actual_fraud > 0:
        for pattern, count in sorted(fraud_counts.items()):
            print(f"  {pattern}: {count} ({count/actual_fraud*100:.1f}%)")
    print("="*70)

    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(transactions, f, indent=2)
    print(f"\n✓ Saved to: {filepath}")

    return transactions, wallets


# ================== V3: TEMPORAL FEATURE ENGINEERING ==================
def engineer_temporal_features(transactions, threat_list):
    """
    V3: Engineer features TEMPORALLY to prevent data leakage.
    This function calculates all 26 features realistically.
    """
    print("\n=== V3: Engineering Features (Temporal, 26 Features, No Leakage) ===")
    
    threat_set = set(w.lower() for w in threat_list)

    # Historical Wallet Profiles
    wallet_profiles = defaultdict(lambda: {
        'tx_count': 0, 'total_sent': 0.0, 'recipients': set(),
        'total_time_diff': 0.0, 'min_time_diff': float('inf'),
        'last_tx_time': None, 'first_tx_time': None,
    })
    receiver_profiles = defaultdict(lambda: {
        'tx_count': 0, 'total_received': 0.0, 'senders': set(),
    })
    
    features = []
    labels = []
    print(f"Processing {len(transactions)} transactions temporally...")

    for i, tx in enumerate(transactions):
        if i % 25000 == 0 and i > 0:
            print(f"  Processed {i}/{len(transactions)}...")
            
        from_addr = tx['from_address'].lower()
        to_addr = tx['to_address'].lower()
        value = tx['value_eth']
        gas = tx['gas_price']
        try:
            current_time = datetime.fromisoformat(tx['timestamp'].replace('Z', ''))
        except:
            current_time = START_DATE
            
        # 1. GET historical features *before* updating
        sender_profile = wallet_profiles[from_addr]
        receiver_profile = receiver_profiles[to_addr]

        # 2. Build the 26-feature vector
        feat = []
        
        # Basic features (0-2)
        feat.extend([value, gas, value * gas])

        # Threat intelligence (3-4)
        feat.extend([1 if from_addr in threat_set else 0, 1 if to_addr in threat_set else 0])

        # Sender profile features (5-9)
        sender_tx_count = sender_profile['tx_count']
        sender_avg_sent = (sender_profile['total_sent'] / max(1, sender_tx_count)) if sender_tx_count > 0 else 0.0
        sender_unique_recipients = len(sender_profile['recipients'])
        recipient_diversity = sender_unique_recipients / max(1, sender_tx_count)
        feat.extend([
            sender_tx_count,                        # 5: Sender tx count
            sender_avg_sent,                        # 6: Sender avg value
            value / max(sender_avg_sent, 0.001),    # 7: Value deviation ratio
            sender_unique_recipients,               # 8: Sender unique recipients
            recipient_diversity,                    # 9: Recipient diversity
        ])
        
        # Receiver profile features (10-12)
        feat.extend([
            receiver_profile['tx_count'],           # 10: Receiver tx count
            receiver_profile['total_received'],     # 11: Receiver total received
            len(receiver_profile['senders']),       # 12: Receiver unique senders
        ])

        # Temporal features (13-17)
        hour, weekday = current_time.hour, current_time.weekday()
        odd_hours = 1 if 2 <= hour <= 5 else 0
        
        if sender_profile['last_tx_time']:
            time_since_last = (current_time - sender_profile['last_tx_time']).total_seconds()
            avg_time_between_tx = sender_profile['total_time_diff'] / max(1, sender_tx_count - 1) if sender_tx_count > 1 else 86400.0
            min_time_between_tx = sender_profile['min_time_diff'] if sender_profile['min_time_diff'] != float('inf') else 86400.0
        else:
            avg_time_between_tx = 86400.0
            min_time_between_tx = 86400.0
        feat.extend([
            hour,                 # 13: Hour of day
            weekday,              # 14: Day of week
            odd_hours,            # 15: Abnormal timing (2-5 AM)
            avg_time_between_tx,  # 16: Avg time between sent txs
            min_time_between_tx   # 17: Min time between sent txs
        ])
        
        # Pattern-specific features (18-25)
        feat.extend([
            1 if value > 100 else 0,             # 18: Very high value
            1 if value > 50 else 0,              # 19: High value
            1 if value < 0.1 else 0,             # 20: Micro value
            1 if gas > 150 else 0,               # 21: High gas
            1 if gas < 30 else 0,                # 22: Low gas
            gas / max(value, 0.001),             # 23: Gas-to-value ratio
            1 if from_addr == to_addr else 0,     # 24: Self-transfer
            1 if (value > 0 and (value * 100) % 100 < 0.01) else 0, # 25: Round number
        ])

        if len(feat) != 26:
            raise Exception(f"Feature engineering created {len(feat)} features, expected 26.")

        features.append(feat)
        labels.append(tx['is_fraud'])

        # 3. UPDATE profiles *after* feature creation
        # Sender
        sender_profile['tx_count'] += 1
        sender_profile['total_sent'] += value
        sender_profile['recipients'].add(to_addr)
        if sender_profile['last_tx_time']:
            time_diff = (current_time - sender_profile['last_tx_time']).total_seconds()
            sender_profile['total_time_diff'] += time_diff
            sender_profile['min_time_diff'] = min(sender_profile['min_time_diff'], time_diff)
        else:
            sender_profile['first_tx_time'] = current_time
        sender_profile['last_tx_time'] = current_time
        
        # Receiver
        receiver_profiles[to_addr]['tx_count'] += 1
        receiver_profiles[to_addr]['total_received'] += value
        receiver_profiles[to_addr]['senders'].add(from_addr)

    print(f"✓ Engineered {len(features[0])} features per transaction (Temporally)")
    
    # Combine profiles for DB creation
    final_profiles = {}
    for addr, data in wallet_profiles.items():
        final_profiles[addr] = {
            'avg_sent': (data['total_sent'] / max(1, data['tx_count'])) if data['tx_count'] > 0 else 0.0,
            'tx_count_sent': data['tx_count'], 'unique_recipients': len(data['recipients']),
            'avg_time_between_tx': (data['total_time_diff'] / max(1, data['tx_count'] - 1)) if data['tx_count'] > 1 else 86400.0,
            'min_time_between_tx': data['min_time_diff'] if data['min_time_diff'] != float('inf') else 86400.0,
            'first_seen': data['first_tx_time'].isoformat() + 'Z' if data['first_tx_time'] else None,
            'last_seen': data['last_tx_time'].isoformat() + 'Z' if data['last_tx_time'] else None,
            'tx_count_received': 0, 'total_received': 0.0, 'unique_senders': 0,
        }
    for addr, data in receiver_profiles.items():
        if addr not in final_profiles:
            final_profiles[addr] = {
                'avg_sent': 0.0, 'tx_count_sent': 0, 'unique_recipients': 0,
                'avg_time_between_tx': 86400.0, 'min_time_between_tx': 86400.0,
                'first_seen': None, 'last_seen': None,
            }
        final_profiles[addr]['tx_count_received'] = data['tx_count']
        final_profiles[addr]['total_received'] = data['total_received']
        final_profiles[addr]['unique_senders'] = len(data['senders'])
        
    print("✓ Final wallet profiles calculated for DB.")
    return np.array(features), np.array(labels), final_profiles


# ================== DB & FILE CREATION ==================
def create_directories():
    """Create necessary directories"""
    for directory in [PROJECT_ROOT / 'data', PROJECT_ROOT / 'models', PROJECT_ROOT / 'output']:
        os.makedirs(directory, exist_ok=True)

def create_wallet_profiles_db(all_wallets, final_profiles):
    """V3: Create SQLite database with REAL wallet profiles from training data."""
    print("\n=== V3: Creating Wallet Profiles Database (from REAL data) ===")
    filepath = PROJECT_ROOT / 'data' / 'wallet_profiles.db'

    try:
        conn = sqlite3.connect(str(filepath))
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS wallet_profiles")
        cursor.execute("""
            CREATE TABLE wallet_profiles (
                wallet_address TEXT PRIMARY KEY,
                avg_sent_value REAL, tx_count_sent INTEGER, unique_recipients INTEGER, 
                avg_time_between_tx REAL, min_time_between_tx REAL,
                tx_count_received INTEGER, total_received REAL, unique_senders INTEGER,
                first_seen TEXT, last_seen TEXT
            )
        """)
        
        print(f"Populating profiles for {len(all_wallets)} wallets...")
        wallets_to_insert = []
        for wallet in all_wallets:
            wallet_lower = wallet.lower()
            profile = final_profiles.get(wallet_lower, {})
            wallets_to_insert.append((
                wallet_lower,
                profile.get('avg_sent', 0.0), profile.get('tx_count_sent', 0),
                profile.get('unique_recipients', 0), profile.get('avg_time_between_tx', 86400.0),
                profile.get('min_time_between_tx', 86400.0), profile.get('tx_count_received', 0),
                profile.get('total_received', 0.0), profile.get('unique_senders', 0),
                profile.get('first_seen'), profile.get('last_seen')
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO wallet_profiles
            (wallet_address, avg_sent_value, tx_count_sent, unique_recipients, 
             avg_time_between_tx, min_time_between_tx, tx_count_received, 
             total_received, unique_senders, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, wallets_to_insert)

        conn.commit()
        count = cursor.execute("SELECT COUNT(*) FROM wallet_profiles").fetchone()[0]
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