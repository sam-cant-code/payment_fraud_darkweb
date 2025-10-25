"""
V4.0 - Utility Script for Data Generation & Feature Engineering
- Dynamic wallet generation (using UUIDs)
- More realistic fraud patterns with noise
- V4 temporal feature engineering (NO threat list leakage)
"""

import json
import sqlite3
import random
import os
import math
import numpy as np
import uuid # For dynamic wallets
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque

# ================== CONFIGURATION ==================
# Correctly define PROJECT_ROOT once at the top
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    BACKEND_DIR = SCRIPT_DIR.parent
    PROJECT_ROOT = BACKEND_DIR.parent
except NameError:
    # Fallback for interactive environments
    SCRIPT_DIR = Path.cwd()
    BACKEND_DIR = SCRIPT_DIR
    PROJECT_ROOT = BACKEND_DIR.parent
    
# Add backend/scripts to sys.path
sys.path.append(str(SCRIPT_DIR))


# Dataset configuration
TOTAL_TRANSACTIONS = 250000
FRAUD_RATE = 0.03
NUM_FRAUD = int(TOTAL_TRANSACTIONS * FRAUD_RATE)
NUM_NORMAL = TOTAL_TRANSACTIONS - NUM_FRAUD

# Wallet pools - Sizes remain the same
NUM_MALICIOUS_WALLETS = 100
NUM_MIXER_WALLETS = 10
NUM_NORMAL_WALLETS = 1000
NUM_HIGH_VALUE_WALLETS = 50

# Fraud pattern distribution (Adjusted for noise and hybrid)
FRAUD_PATTERNS = {
    'malicious_interaction': int(NUM_FRAUD * 0.25),
    'high_value_suspicious': int(NUM_FRAUD * 0.15),
    'mixer_usage': int(NUM_FRAUD * 0.20),
    'rapid_micro_transfers': int(NUM_FRAUD * 0.10),
    'realistic_circular_transfers': int(NUM_FRAUD * 0.10),
    'abnormal_timing': int(NUM_FRAUD * 0.10),
    'self_transfers': int(NUM_FRAUD * 0.05),
    'hybrid_fraud': int(NUM_FRAUD * 0.05),
}
# Adjust if sum doesn't match NUM_FRAUD
total_assigned = sum(FRAUD_PATTERNS.values())
diff = NUM_FRAUD - total_assigned
if diff != 0:
    # Add/remove difference to the most common type
    FRAUD_PATTERNS['malicious_interaction'] = FRAUD_PATTERNS.get('malicious_interaction', 0) + diff
    # Filter out any zero counts if adjustment made it zero
    FRAUD_PATTERNS = {k: v for k, v in FRAUD_PATTERNS.items() if v > 0}

# Temporal configuration
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 10, 1, tzinfo=timezone.utc)
TOTAL_SECONDS = (END_DATE - START_DATE).total_seconds()


# ================== DYNAMIC WALLET GENERATION ==================
def generate_dynamic_wallet_pools():
    """
    Generate different wallet addresses for each run using UUIDs.
    Ensures the model doesn't memorize specific addresses.
    """
    print("\n=== Generating Dynamic Wallet Pools (V4) ===")

    # Use UUID4 for randomness and slice to get 40 hex chars (like ETH address)
    malicious = {f"0x{uuid.uuid4().hex[:40]}" for _ in range(NUM_MALICIOUS_WALLETS)}
    mixers = {f"0x{uuid.uuid4().hex[:40]}" for _ in range(NUM_MIXER_WALLETS)}
    normal = {f"0x{uuid.uuid4().hex[:40]}" for _ in range(NUM_NORMAL_WALLETS)}
    high_value = {f"0x{uuid.uuid4().hex[:40]}" for _ in range(NUM_HIGH_VALUE_WALLETS)}

    # Ensure no accidental overlap (highly unlikely with UUIDs)
    all_wallets_set = malicious | mixers | normal | high_value
    expected_count = NUM_MALICIOUS_WALLETS + NUM_MIXER_WALLETS + NUM_NORMAL_WALLETS + NUM_HIGH_VALUE_WALLETS
    if len(all_wallets_set) != expected_count:
        print("Warning: Potential wallet overlap detected, though unlikely.")

    print(f"✓ Created {len(malicious)} dynamic malicious wallets")
    print(f"✓ Created {len(mixers)} dynamic mixer wallets")
    print(f"✓ Created {len(normal)} dynamic normal wallets")
    print(f"✓ Created {len(high_value)} dynamic high-value legitimate wallets")

    # Threat list still includes malicious + mixers for potential runtime use (Agent 1)
    # but WILL NOT be used in V4 feature engineering for the ML model.
    threat_list = list(malicious | mixers)

    return {
        'malicious': list(malicious),
        'mixers': list(mixers),
        'normal': list(normal),
        'high_value': list(high_value),
        'threat_list': threat_list,
        'all_wallets': list(all_wallets_set)
    }


# ================== REALISTIC VALUE DISTRIBUTIONS ==================
def get_normal_value():
    return max(0.001, min(random.lognormvariate(math.log(0.5), 1.2), 50.0))
def get_high_value(): # Legitimate high value
    return random.uniform(50, 500)
def get_suspicious_value(): # Fraudulent high value (often slightly different range/pattern)
    return random.uniform(80, 300)
def get_micro_value():
    return random.uniform(0.001, 0.05)
def get_normal_gas():
    return round(random.uniform(20, 100), 2)
def get_high_gas():
    return round(random.uniform(150, 500), 2)

def generate_timestamp(base_time=None, max_seconds_offset=None):
    if base_time is None:
        base_time = START_DATE
    if max_seconds_offset is None:
        max_seconds_offset = TOTAL_SECONDS
    
    offset_seconds = random.uniform(0, max_seconds_offset)
    timestamp = base_time + timedelta(seconds=offset_seconds)
    
    # Ensure timestamp doesn't exceed END_DATE
    timestamp = min(timestamp, END_DATE)
    
    return timestamp.isoformat().replace('+00:00', 'Z')


# ================== V4 FRAUD PATTERN GENERATORS (with Noise) ==================

# --- Base Transaction Generator ---
def create_base_tx(from_addr, to_addr, value, gas, timestamp, is_fraud, fraud_type):
    return {
        "tx_hash": f"0x{uuid.uuid4().hex[:64]}", # Use UUID for hash too
        "from_address": from_addr,
        "to_address": to_addr,
        "value_eth": round(max(0.000001, value), 6), # Ensure non-zero positive
        "gas_price": round(max(1.0, gas), 2),
        "timestamp": timestamp,
        "is_fraud": is_fraud,
        "fraud_type": fraud_type
    }

# --- Normal Transaction (Can be used as noise) ---
def generate_normal_transaction(wallets, timestamp):
    """Generates a standard, legitimate transaction."""
    use_high_value_pool = random.random() < 0.1 # Chance to use a whale wallet
    
    if use_high_value_pool:
        from_wallet = random.choice(wallets['high_value'])
        value = get_high_value() if random.random() < 0.2 else get_normal_value() # Whales can send normal amounts too
    else:
        from_wallet = random.choice(wallets['normal'])
        value = get_normal_value()

    to_wallet = random.choice(wallets['normal'])
    # Avoid self-transfers for standard normal tx, unless specifically testing that
    while from_wallet == to_wallet:
        to_wallet = random.choice(wallets['normal'])
        
    gas = get_normal_gas() * random.uniform(0.9, 1.1) # Slight gas variation
    
    return create_base_tx(from_wallet, to_wallet, value, gas, timestamp, 0, "normal")


# --- Legitimate High Value / DeFi (Can be confused with fraud) ---
def generate_legitimate_defi_transaction(wallets, timestamp):
     """Large legitimate DeFi swap or high-value transfer."""
     from_wallet = random.choice(wallets['high_value'])
     # Simulate interaction with a known contract or another high-value wallet
     to_wallet = random.choice(["0xDefi_Pool_1", "0xExchange_Addr", random.choice(wallets['high_value'])])
     value = random.uniform(50, 500)
     gas = random.uniform(80, 150) # Higher gas usually for contract interactions
     
     return create_base_tx(from_wallet, to_wallet, value, gas, timestamp, 0, "legitimate_defi")
     
# --- Malicious Interaction (Modified slightly) ---
def generate_malicious_interaction(wallets, timestamp):
    """Interaction involving a known malicious wallet."""
    use_malicious_sender = random.random() < 0.6
    from_wallet = random.choice(wallets['malicious']) if use_malicious_sender else random.choice(wallets['normal'])
    to_wallet = random.choice(wallets['normal']) if use_malicious_sender else random.choice(wallets['malicious'])
    
    # Value can be normal or suspicious, sometimes round numbers
    if random.random() < 0.1:
        value = random.choice([10.0, 50.0, 100.0])
    else:
         value = get_normal_value() if random.random() < 0.6 else get_suspicious_value()
         
    gas = get_normal_gas() * random.uniform(0.8, 1.3) # Slightly wider gas variation
    
    return create_base_tx(from_wallet, to_wallet, value, gas, timestamp, 1, "malicious_interaction")

# --- High Value Suspicious ---
def generate_high_value_suspicious(wallets, timestamp):
    """High value tx between normal wallets, potentially suspicious pattern."""
    from_wallet = random.choice(wallets['normal'])
    to_wallet = random.choice(wallets['normal'])
    while from_wallet == to_wallet:
        to_wallet = random.choice(wallets['normal'])
        
    value = get_suspicious_value()
    # Sometimes uses unusually low or high gas
    gas = get_normal_gas() * random.uniform(0.5, 1.5) if random.random() < 0.8 else random.choice([get_high_gas(), random.uniform(5, 15)])
    
    return create_base_tx(from_wallet, to_wallet, value, gas, timestamp, 1, "high_value_suspicious")

# --- Mixer Usage ---
def generate_mixer_usage(wallets, timestamp):
    """Transaction involving a known mixer address."""
    to_mixer = random.random() < 0.7
    from_wallet = random.choice(wallets['normal']) if to_mixer else random.choice(wallets['mixers'])
    to_wallet = random.choice(wallets['mixers']) if to_mixer else random.choice(wallets['normal'])
    
    value = get_normal_value() * random.uniform(0.7, 1.5) # Vary value slightly
    gas = get_normal_gas()
    
    return create_base_tx(from_wallet, to_wallet, value, gas, timestamp, 1, "mixer_usage")

# --- Rapid Micro Transfers (Added Noise) ---
def generate_rapid_micro_transfers(wallets, base_timestamp_str):
    """Smurfing pattern with added noise."""
    sender = random.choice(wallets['normal'])
    num_fraud_transfers = random.randint(5, 15)
    num_noise_transfers = random.randint(0, 5) # Add 0-5 legit txs from same sender
    
    all_txs = []
    
    base_time = datetime.fromisoformat(base_timestamp_str.replace('Z', ''))
    
    # Generate fraud txs
    current_time_fraud = base_time
    for _ in range(num_fraud_transfers):
        # Shorter, more varied time delays
        current_time_fraud += timedelta(seconds=random.randint(10, 180))
        value = get_micro_value() * random.uniform(0.9, 1.1) # Slight value variation
        to_wallet = random.choice(wallets['normal'])
        while to_wallet == sender: to_wallet = random.choice(wallets['normal'])
        gas = get_normal_gas() * random.uniform(0.95, 1.05)
        
        all_txs.append(create_base_tx(sender, to_wallet, value, gas, current_time_fraud.isoformat().replace('+00:00', 'Z'), 1, "rapid_micro_transfers"))
        
    # Generate noise txs (normal value, interspersed)
    current_time_noise = base_time + timedelta(seconds=random.randint(5, 60)) # Start noise slightly after
    for _ in range(num_noise_transfers):
         current_time_noise += timedelta(seconds=random.randint(60, 600)) # Longer delays for normal
         noise_tx = generate_normal_transaction(wallets, current_time_noise.isoformat().replace('+00:00', 'Z'))
         noise_tx['from_address'] = sender # Make sure noise comes from the same sender
         while noise_tx['to_address'] == sender: noise_tx['to_address'] = random.choice(wallets['normal'])
         all_txs.append(noise_tx)
         
    random.shuffle(all_txs) # Mix fraud and noise
    return all_txs


# --- REALISTIC Circular Transfers (Added Noise) ---
def generate_realistic_circular_transfer(wallets, base_timestamp_str):
    """More realistic circular pattern with noise & variations."""
    chain_length = random.randint(3, 8)  # Slightly longer chains possible
    # Use mostly normal wallets, maybe one high_value occasionally
    pool = wallets['normal'] + wallets['high_value']
    chain_wallets = random.sample(pool, chain_length)

    transfers = []
    noise_txs = []

    base_time = datetime.fromisoformat(base_timestamp_str.replace('Z', ''))
    current_time = base_time

    # Initial value can be higher
    initial_value = get_normal_value() * random.uniform(1.0, 3.0)

    for i in range(chain_length):
        # Vary the time delays much more - hours or even days
        time_delta_hours = random.uniform(0.1, 48) # 6 mins to 2 days
        current_time += timedelta(hours=time_delta_hours)

        from_wallet = chain_wallets[i]
        to_wallet = chain_wallets[(i + 1) % chain_length] # Wraps around for the last step

        # Add more significant value decay/variation at each hop
        # Value might decrease, but not perfectly predictably
        value_multiplier = random.uniform(0.75, 0.98) # Lose 2-25% per hop
        value = initial_value * (value_multiplier ** i) * random.uniform(0.95, 1.05) # Add small fluctuation
        value = max(0.000001, value) # Ensure positive

        gas = get_normal_gas() * random.uniform(0.8, 1.2)

        transfers.append(create_base_tx(from_wallet, to_wallet, value, gas, current_time.isoformat().replace('+00:00', 'Z'), 1, "circular_transfer"))

        # ~30% chance to add a legitimate transaction as noise between hops
        # This noise transaction could be from/to anyone, not necessarily related to the chain
        if random.random() < 0.3:
            noise_time = current_time + timedelta(minutes=random.uniform(1, time_delta_hours * 30)) # Noise happens sometime between hops
            noise_tx = generate_normal_transaction(wallets, noise_time.isoformat().replace('+00:00', 'Z'))
            noise_txs.append(noise_tx)
            
    # Mix fraud and noise transactions
    all_txs = transfers + noise_txs
    # Sort by timestamp BEFORE returning to maintain temporal order for feature engineering
    all_txs.sort(key=lambda x: x['timestamp'])
    return all_txs


# --- Abnormal Timing (Modified for more variation) ---
def generate_abnormal_timing(wallets, base_timestamp_str):
    """Generate a burst of transactions at unusual times (e.g., 2-5 AM UTC)."""
    base_time = datetime.fromisoformat(base_timestamp_str.replace('Z', ''))
    
    # Set time to an odd hour
    burst_start_hour = random.randint(2, 5)
    burst_time = base_time.replace(hour=burst_start_hour, minute=random.randint(0, 59), second=random.randint(0, 59))
    
    num_txs = random.randint(6, 15) # Slightly wider range
    transfers = []
    
    senders = random.sample(wallets['normal'], k=random.randint(1, 3)) # 1-3 senders involved
    receivers = random.sample(wallets['normal'], k=num_txs) # Many receivers
    
    current_time = burst_time
    for i in range(num_txs):
        # More variation in time between txs
        current_time += timedelta(seconds=random.uniform(5, 240)) # 5 seconds to 4 mins
        from_wallet = random.choice(senders)
        to_wallet = receivers[i]
        value = get_normal_value() * random.uniform(0.7, 1.3)
        gas = get_normal_gas() * random.uniform(0.9, 1.1)
        
        transfers.append(create_base_tx(from_wallet, to_wallet, value, gas, current_time.isoformat().replace('+00:00', 'Z'), 1, "abnormal_timing"))
        
    return transfers

# --- Self Transfers ---
def generate_self_transfer(wallets, timestamp):
    """Transaction from a wallet to itself."""
    wallet = random.choice(wallets['normal'])
    value = get_normal_value() * random.uniform(0.5, 2.0)
    # Gas might be slightly higher for self-transfers sometimes
    gas = get_normal_gas() * random.uniform(1.0, 1.5)
    
    return create_base_tx(wallet, wallet, value, gas, timestamp, 1, "self_transfer")
    
# --- V4: Hybrid Fraud Pattern ---
def generate_hybrid_fraud(wallets, base_timestamp_str):
    """Combines multiple techniques, e.g., Mixer -> Delay -> Micro-transfers."""
    all_txs = []
    base_time = datetime.fromisoformat(base_timestamp_str.replace('Z', ''))
    
    # Step 1: Send funds TO a mixer
    mixer_in_time = base_time + timedelta(minutes=random.uniform(1, 60))
    mixer_in_tx = generate_mixer_usage(wallets, mixer_in_time.isoformat().replace('+00:00', 'Z'))
    # Ensure it's going TO the mixer
    while mixer_in_tx['to_address'] not in wallets['mixers']:
        mixer_in_tx = generate_mixer_usage(wallets, mixer_in_time.isoformat().replace('+00:00', 'Z'))
    all_txs.append(mixer_in_tx)
    
    # The 'source' wallet for the smurfing will be different from the one that sent to the mixer
    smurf_sender = random.choice(wallets['normal'])
    
    # Step 2: Delay (Simulates funds moving through mixer)
    delay_hours = random.uniform(1, 12) # 1 to 12 hour delay
    smurf_start_time = mixer_in_time + timedelta(hours=delay_hours)
    
    # Step 3: Rapid micro-transfers (smurfing) FROM a different wallet
    smurf_txs = generate_rapid_micro_transfers(wallets, smurf_start_time.isoformat().replace('+00:00', 'Z'))
    # Make sure the smurf txs use the designated smurf_sender
    for tx in smurf_txs:
        tx['from_address'] = smurf_sender
        tx['fraud_type'] = 'hybrid_mixer_smurf' # More specific type
        
    all_txs.extend(smurf_txs)
    
    # Sort by timestamp
    all_txs.sort(key=lambda x: x['timestamp'])
    return all_txs


# ================== MAIN DATA GENERATION (V4) ==================
def create_enhanced_mock_transactions(load_if_exists=False):
    """
    V4: Create comprehensive fraud detection dataset with dynamic wallets and noisy patterns.
    If load_if_exists=True, it will load from file instead of regenerating.
    """
    print("\n" + "="*70)
    print("V4: ENHANCED MOCK TRANSACTION GENERATION")
    print("(Dynamic Wallets, Realistic Patterns)")
    print("="*70)

    filepath = PROJECT_ROOT / 'data' / 'mock_blockchain_transactions.json'
    # Use DYNAMIC wallets
    wallets = generate_dynamic_wallet_pools()

    if load_if_exists and filepath.exists():
        try:
            print(f"Loading existing dataset from {filepath}...")
            with open(filepath, 'r') as f:
                transactions = json.load(f)
            print(f"✓ Loaded {len(transactions)} transactions.")
            # We still return the newly generated DYNAMIC wallets, even if loading old tx data
            return transactions, wallets
        except Exception as e:
            print(f"Error loading {filepath}: {e}. Regenerating...")

    print(f"Generating {TOTAL_TRANSACTIONS} new transactions...")
    print(f"Target fraud rate: {FRAUD_RATE*100:.1f}% ({NUM_FRAUD} fraud, {NUM_NORMAL} normal)")
    print("Fraud pattern targets:")
    for pattern, count in FRAUD_PATTERNS.items():
        print(f"  - {pattern}: {count}")
    print("="*70)

    transactions = []
    fraud_counts = defaultdict(int)
    generated_fraud_count = 0

    # Generate Fraud Transactions (using V4 functions)
    print("\n=== Generating Fraud Patterns (V4) ===")
    
    fraud_generators = {
        'malicious_interaction': generate_malicious_interaction,
        'high_value_suspicious': generate_high_value_suspicious,
        'mixer_usage': generate_mixer_usage,
        'rapid_micro_transfers': generate_rapid_micro_transfers,
        'realistic_circular_transfers': generate_realistic_circular_transfer,
        'abnormal_timing': generate_abnormal_timing,
        'self_transfers': generate_self_transfer,
        'hybrid_fraud': generate_hybrid_fraud,
    }

    # Generate fraud based on FRAUD_PATTERNS counts
    # This loop generates roughly the target number, handling multi-transaction patterns
    for fraud_type, target_count in FRAUD_PATTERNS.items():
        generator_func = fraud_generators.get(fraud_type)
        if not generator_func:
            print(f"Warning: No generator function found for fraud type '{fraud_type}'")
            continue
            
        print(f"Generating approx {target_count} transactions for '{fraud_type}'...")
        # Estimate how many times to call the generator
        # Generators producing lists need fewer calls
        calls_needed = target_count
        if fraud_type in ['rapid_micro_transfers', 'realistic_circular_transfers', 'abnormal_timing', 'hybrid_fraud']:
             calls_needed = max(1, target_count // 8) # Estimate avg 8 tx per call

        for i in range(calls_needed):
            if generated_fraud_count >= NUM_FRAUD * 1.1: # Stop if we overshoot too much
                 break
            
            ts = generate_timestamp()
            result = generator_func(wallets, ts)
            
            if isinstance(result, list):
                transactions.extend(result)
                count_added = len(result)
                # Count fraud type based on the first element's type (might be hybrid)
                if result: fraud_counts[result[0]['fraud_type']] += count_added
            elif isinstance(result, dict):
                transactions.append(result)
                count_added = 1
                fraud_counts[result['fraud_type']] += count_added
            else:
                count_added = 0
                
            generated_fraud_count += count_added
                 
        if generated_fraud_count >= NUM_FRAUD * 1.1: break # Break outer loop too

    # Generate Normal & Legitimate High Value Transactions
    num_normal_needed = TOTAL_TRANSACTIONS - len(transactions)
    print(f"\n=== Generating {num_normal_needed} Normal & Legitimate High Value Transactions ===")
    for i in range(num_normal_needed):
        if i % 25000 == 0 and i > 0:
            print(f"  Generated {i}/{num_normal_needed}...")
        
        ts = generate_timestamp()
        if random.random() < 0.02: # 2% chance of being a legit DeFi/Whale tx
            tx = generate_legitimate_defi_transaction(wallets, ts)
        else:
            tx = generate_normal_transaction(wallets, ts)
        transactions.append(tx)

    print("\nSorting all transactions by timestamp...")
    transactions.sort(key=lambda x: x['timestamp'])

    # Final dataset summary
    print("\n" + "="*70)
    print("V4 DATASET SUMMARY")
    print("="*70)
    actual_normal = len([t for t in transactions if t.get('is_fraud', 0) == 0])
    actual_fraud = len([t for t in transactions if t.get('is_fraud', 0) == 1])
    total_generated = len(transactions)
    print(f"Total Generated: {total_generated}")
    print(f"  Normal:      {actual_normal} ({actual_normal/total_generated*100:.2f}%)")
    print(f"  Fraud:       {actual_fraud} ({actual_fraud/total_generated*100:.2f}%)")
    
    print(f"\nActual Fraud Pattern Breakdown (Counts):")
    final_fraud_type_counts = defaultdict(int)
    for tx in transactions:
        if tx.get('is_fraud', 0) == 1:
            final_fraud_type_counts[tx.get('fraud_type', 'unknown')] += 1
            
    if actual_fraud > 0:
        for pattern, count in sorted(final_fraud_type_counts.items()):
            print(f"  - {pattern}: {count} ({count/actual_fraud*100:.1f}%)")
    else:
        print("  - No fraud transactions generated.")
    print("="*70)

    # Save dataset
    os.makedirs(filepath.parent, exist_ok=True)
    try:
        with open(filepath, 'w') as f:
            json.dump(transactions, f, indent=2)
        print(f"\n✓ Saved V4 dataset to: {filepath}")
    except Exception as e:
         print(f"\nError saving dataset: {e}")

    return transactions, wallets


# ================== V4: FEATURE ENGINEERING ==================
# This is the V4 feature list that matches the training script
FEATURE_NAMES_V4 = [
    # Basic (0-5)
    'value_eth', 'gas_price', 'total_cost', 'log_value', 'log_gas', 'is_nonzero',
    # Sender behavioral (6-14)
    'sender_tx_count', 'log_sender_tx_count', 'sender_avg_value', 'sender_std_value',
    'value_deviation_ratio', 'value_z_score', 'sender_unique_recipients', 'recipient_diversity',
    'sender_is_new',
    # Receiver behavioral (15-21)
    'receiver_tx_count', 'receiver_avg_received', 'receiver_unique_senders',
    'sender_diversity', 'receiver_is_new', 'is_collector', 'is_distributor',
    # Temporal (22-27)
    'hour_normalized', 'weekday_normalized', 'graveyard_shift', 'is_weekend',
    'log_avg_time_between', 'log_min_time_between',
    # Relationship (28-31)
    'sender_to_receiver_history', 'known_relationship', 'self_transfer', 'address_similarity',
    # Anomaly (32-34)
    'very_high_value', 'micro_value', 'gas_value_ratio',
]

# This is the function that builds the features
def engineer_features_v4(transactions, threat_list_ignored):
    """
    V4: Engineer features TEMPORALLY based on the 35 features.
    - PREVENTS data leakage by calculating historically.
    - DOES NOT use threat list information for ML features.
    """
    print("\n=== V4: Engineering Features (Temporal, 35 Features, No Leakage) ===")

    # Wallets track more detailed history now
    sender_profiles = defaultdict(lambda: {
        'tx_timestamps': deque(maxlen=50), # Track last 50 timestamps
        'sent_values': deque(maxlen=50),   # Track last 50 sent values
        'recipients': defaultdict(int),     # Count txs per recipient
        'first_tx_time': None,
        'tx_count': 0,
    })
    receiver_profiles = defaultdict(lambda: {
        'senders': defaultdict(int),        # Count txs per sender
        'received_values': deque(maxlen=50),# Track last 50 received values
        'first_tx_time': None,
        'tx_count': 0,
    })
    # Track interactions between pairs
    interaction_history = defaultdict(lambda: {'count': 0, 'last_time': None})

    features = []
    labels = []
    print(f"Processing {len(transactions)} transactions temporally for V4 features...")

    # Small value epsilon to avoid division by zero or log(0)
    epsilon = 1e-9

    for i, tx in enumerate(transactions):
        if i % 25000 == 0 and i > 0:
            print(f"  Processed {i}/{len(transactions)}...")

        # Transaction data
        from_addr = tx['from_address'].lower()
        to_addr = tx['to_address'].lower()
        value = float(tx['value_eth'])
        gas = float(tx['gas_price'])
        try:
            current_time = datetime.fromisoformat(tx['timestamp'].replace('Z', '')).replace(tzinfo=timezone.utc)
        except ValueError:
            # Fallback if timestamp is invalid
            current_time = START_DATE + timedelta(seconds=(i / len(transactions)) * TOTAL_SECONDS)

        # 1. GET historical features *before* updating profiles
        sender = sender_profiles[from_addr]
        receiver = receiver_profiles[to_addr]
        interaction_key = tuple(sorted((from_addr, to_addr))) # Consistent key for interaction pair
        interaction = interaction_history[interaction_key]

        # 2. Build the V4 feature vector (35 features)
        feat = []

        # --- Basic Features (0-5) ---
        total_cost = value * gas # Simple interaction
        log_value = math.log10(value + epsilon)
        log_gas = math.log10(gas + epsilon)
        is_nonzero = 1 if value > epsilon else 0
        feat.extend([value, gas, total_cost, log_value, log_gas, is_nonzero])

        # --- Sender Behavioral Features (6-14) ---
        sender_tx_count = sender['tx_count']
        log_sender_tx_count = math.log10(sender_tx_count + 1) # Log of count + 1

        sender_avg_value = np.mean(list(sender['sent_values'])) if sender['sent_values'] else 0
        sender_std_value = np.std(list(sender['sent_values'])) if len(sender['sent_values']) > 1 else 0

        # Deviation compared to sender's average
        value_deviation_ratio = value / (sender_avg_value + epsilon)
        # Z-score: How many std deviations away from the mean
        value_z_score = (value - sender_avg_value) / (sender_std_value + epsilon) if sender_std_value > epsilon else 0

        sender_unique_recipients = len(sender['recipients'])
        # Recipient diversity: unique recipients / total txs sent
        recipient_diversity = sender_unique_recipients / (sender_tx_count + epsilon)

        sender_is_new = 1 if sender_tx_count == 0 else 0 # Is this the sender's first transaction?

        feat.extend([
            sender_tx_count, log_sender_tx_count, sender_avg_value, sender_std_value,
            value_deviation_ratio, value_z_score, sender_unique_recipients, recipient_diversity,
            sender_is_new
        ])

        # --- Receiver Behavioral Features (15-21) ---
        receiver_tx_count = receiver['tx_count']
        receiver_avg_received = np.mean(list(receiver['received_values'])) if receiver['received_values'] else 0
        receiver_unique_senders = len(receiver['senders'])
        # Sender diversity for receiver: unique senders / total txs received
        sender_diversity = receiver_unique_senders / (receiver_tx_count + epsilon)

        receiver_is_new = 1 if receiver_tx_count == 0 else 0

        # Simple heuristics for collector/distributor based on unique counterparts
        is_collector = 1 if receiver_unique_senders > 10 and sender_unique_recipients < 5 else 0
        is_distributor = 1 if sender_unique_recipients > 10 and receiver_unique_senders < 5 else 0

        feat.extend([
            receiver_tx_count, receiver_avg_received, receiver_unique_senders,
            sender_diversity, receiver_is_new, is_collector, is_distributor
        ])

        # --- Temporal Features (22-27) ---
        hour = current_time.hour
        weekday = current_time.weekday() # Monday = 0, Sunday = 6

        # Normalize hour (0-1) and weekday (0-1)
        hour_normalized = hour / 23.0
        weekday_normalized = weekday / 6.0

        graveyard_shift = 1 if 2 <= hour <= 5 else 0 # 2 AM to 5 AM UTC
        is_weekend = 1 if weekday >= 5 else 0 # Saturday or Sunday

        # Time between transactions for sender
        time_diffs = np.diff([ts.timestamp() for ts in sender['tx_timestamps']]) if len(sender['tx_timestamps']) > 1 else []
        avg_time_between = np.mean(time_diffs) if len(time_diffs) > 0 else 3600*24*7 # Default to 1 week if no history
        min_time_between = np.min(time_diffs) if len(time_diffs) > 0 else 3600*24*7

        log_avg_time_between = math.log10(avg_time_between + epsilon)
        log_min_time_between = math.log10(min_time_between + epsilon)

        feat.extend([
            hour_normalized, weekday_normalized, graveyard_shift, is_weekend,
            log_avg_time_between, log_min_time_between
        ])

        # --- Relationship Features (28-31) ---
        sender_to_receiver_history = sender['recipients'].get(to_addr, 0) # How many times sender sent to this receiver
        known_relationship = 1 if interaction['count'] > 0 else 0 # Have these two interacted before?
        self_transfer = 1 if from_addr == to_addr else 0
        
        # Simple address similarity heuristic
        similarity = 0
        if len(from_addr) == 42 and len(to_addr) == 42:
           if from_addr[:8] == to_addr[:8] or from_addr[-6:] == to_addr[-6:]:
               similarity = 1
        address_similarity = similarity

        feat.extend([
            sender_to_receiver_history, known_relationship, self_transfer, address_similarity
        ])

        # --- Anomaly Features (32-34) ---
        very_high_value = 1 if value > 100 else 0 # Flag for exceptionally high values
        micro_value = 1 if value < 0.01 else 0 # Flag for very small values
        gas_value_ratio = gas / (value + epsilon) # Ratio of gas cost to value transferred

        feat.extend([
            very_high_value, micro_value, gas_value_ratio
        ])

        # Final check: Ensure correct number of features
        if len(feat) != len(FEATURE_NAMES_V4):
            raise Exception(f"Feature engineering created {len(feat)} features, expected {len(FEATURE_NAMES_V4)}.")

        features.append(feat)
        labels.append(tx['is_fraud'])

        # 3. UPDATE profiles *after* feature creation
        # Sender Update
        if not sender['first_tx_time']: sender['first_tx_time'] = current_time
        sender['tx_timestamps'].append(current_time)
        sender['sent_values'].append(value)
        sender['recipients'][to_addr] += 1
        sender['tx_count'] += 1

        # Receiver Update
        if not receiver['first_tx_time']: receiver['first_tx_time'] = current_time
        receiver['senders'][from_addr] += 1
        receiver['received_values'].append(value)
        receiver['tx_count'] += 1
        
        # Interaction Update
        interaction['count'] += 1
        interaction['last_time'] = current_time


    print(f"✓ Engineered {len(features[0])} features per transaction for V4 (Temporally)")

    # Combine profiles for DB creation (more comprehensive)
    final_profiles_for_db = {}

    # Process sender profiles
    for addr, data in sender_profiles.items():
        profile = final_profiles_for_db.setdefault(addr.lower(), {})
        profile['tx_count_sent'] = data['tx_count']
        profile['total_sent'] = sum(data['sent_values'])
        profile['avg_sent_value'] = np.mean(list(data['sent_values'])) if data['sent_values'] else 0
        profile['std_sent_value'] = np.std(list(data['sent_values'])) if len(data['sent_values']) > 1 else 0
        profile['unique_recipients'] = len(data['recipients'])
        
        time_diffs_db = np.diff([ts.timestamp() for ts in data['tx_timestamps']]) if len(data['tx_timestamps']) > 1 else []
        profile['avg_time_between_tx'] = float(np.mean(time_diffs_db)) if len(time_diffs_db) > 0 else None
        profile['min_time_between_tx'] = float(np.min(time_diffs_db)) if len(time_diffs_db) > 0 else None
        
        profile['first_seen'] = data['first_tx_time'].isoformat().replace('+00:00', 'Z') if data['first_tx_time'] else None
        # Get the last timestamp from the deque
        profile['last_seen'] = data['tx_timestamps'][-1].isoformat().replace('+00:00', 'Z') if data['tx_timestamps'] else None

    # Process receiver profiles, updating existing entries or adding new ones
    for addr, data in receiver_profiles.items():
        profile = final_profiles_for_db.setdefault(addr.lower(), {})
        profile['tx_count_received'] = data['tx_count']
        profile['total_received'] = sum(data['received_values'])
        profile['avg_received_value'] = np.mean(list(data['received_values'])) if data['received_values'] else 0
        profile['unique_senders'] = len(data['senders'])
        
        # Update first/last seen if receiver info is earlier/later
        first_seen_receiver = data['first_tx_time'].isoformat().replace('+00:00', 'Z') if data['first_tx_time'] else None
        if first_seen_receiver and (not profile.get('first_seen') or first_seen_receiver < profile['first_seen']):
            profile['first_seen'] = first_seen_receiver
            
    print("✓ Final wallet profiles calculated for V4 DB.")
    return np.array(features), np.array(labels), final_profiles_for_db


# ================== DB & FILE CREATION (V4 Schema) ==================
def create_directories():
    """Create necessary directories"""
    for directory in [PROJECT_ROOT / 'data', PROJECT_ROOT / 'models', PROJECT_ROOT / 'output']:
        os.makedirs(directory, exist_ok=True)

def create_wallet_profiles_db(all_wallets, final_profiles):
    """V4: Create SQLite database with detailed wallet profiles from V4 feature engineering."""
    print("\n=== V4: Creating Wallet Profiles Database (from V4 Features) ===")
    filepath = PROJECT_ROOT / 'data' / 'wallet_profiles.db'

    try:
        conn = sqlite3.connect(str(filepath))
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS wallet_profiles")
        # Updated Schema for V4 profiles
        cursor.execute("""
            CREATE TABLE wallet_profiles (
                wallet_address TEXT PRIMARY KEY,
                tx_count_sent INTEGER DEFAULT 0,
                total_sent REAL DEFAULT 0.0,
                avg_sent_value REAL DEFAULT 0.0,
                std_sent_value REAL DEFAULT 0.0,
                unique_recipients INTEGER DEFAULT 0,
                avg_time_between_tx REAL,
                min_time_between_tx REAL,
                tx_count_received INTEGER DEFAULT 0,
                total_received REAL DEFAULT 0.0,
                avg_received_value REAL DEFAULT 0.0,
                unique_senders INTEGER DEFAULT 0,
                first_seen TEXT,
                last_seen TEXT
            )
        """)

        print(f"Populating V4 profiles for {len(all_wallets)} wallets...")
        wallets_to_insert = []
        for wallet in all_wallets:
            wallet_lower = wallet.lower()
            profile = final_profiles.get(wallet_lower, {})
            wallets_to_insert.append((
                wallet_lower,
                profile.get('tx_count_sent', 0),
                profile.get('total_sent', 0.0),
                profile.get('avg_sent_value', 0.0),
                profile.get('std_sent_value', 0.0),
                profile.get('unique_recipients', 0),
                profile.get('avg_time_between_tx'), # Can be None
                profile.get('min_time_between_tx'), # Can be None
                profile.get('tx_count_received', 0),
                profile.get('total_received', 0.0),
                profile.get('avg_received_value', 0.0),
                profile.get('unique_senders', 0),
                profile.get('first_seen'), # Can be None
                profile.get('last_seen')   # Can be None
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO wallet_profiles
            (wallet_address, tx_count_sent, total_sent, avg_sent_value, std_sent_value,
             unique_recipients, avg_time_between_tx, min_time_between_tx,
             tx_count_received, total_received, avg_received_value, unique_senders,
             first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, wallets_to_insert)

        conn.commit()
        count = cursor.execute("SELECT COUNT(*) FROM wallet_profiles").fetchone()[0]
        print(f"✓ Created V4 wallet profiles DB: {count} wallets")
        print(f"  Saved to: {filepath}")
        conn.close()
    except Exception as e:
        print(f"Error creating V4 database: {e}")


def create_threat_list_file(threat_list):
    """Create dark web wallet threat intelligence file (unchanged)"""
    print("\n=== Creating Threat List File ===")
    filepath = PROJECT_ROOT / 'data' / 'dark_web_wallets.txt'
    try:
        # Ensure threat_list contains only strings before sorting
        valid_threats = [str(wallet).lower() for wallet in threat_list if isinstance(wallet, str) and wallet]
        with open(filepath, 'w') as f:
            for wallet in sorted(valid_threats):
                f.write(f"{wallet}\n")
        print(f"✓ Created threat list with {len(valid_threats)} addresses")
        print(f"  Saved to: {filepath}")
    except Exception as e:
        print(f"Error creating threat list: {e}")

# Example of how to call this if needed (e.g., in initialize_and_train.py)
if __name__ == "__main__":
    print("Running data_and_features.py directly to generate data (for testing)")
    create_directories()
    # Generate data with dynamic wallets and noise
    transactions, wallets = create_enhanced_mock_transactions(load_if_exists=False)
    # Engineer V4 features
    X, y, final_profiles = engineer_features_v4(transactions, wallets['threat_list']) # Pass threat list but it won't be used for features
    # Create DB and threat list file
    create_threat_list_file(wallets['threat_list'])
    create_wallet_profiles_db(wallets['all_wallets'], final_profiles)
    print("\nTest data generation and feature engineering complete.")