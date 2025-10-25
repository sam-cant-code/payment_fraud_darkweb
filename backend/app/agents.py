"""
IMPROVED Agent Logic Module (V4.0)
Changes:
- Agent 2 now uses 35 features (NO threat list)
- Better feature engineering matching training
- Improved ML prediction logic
"""

import pickle
import sqlite3
import json
import sys
from typing import Dict, Optional
import os
from datetime import datetime, timedelta, UTC
import numpy as np
from pathlib import Path

# --- PATH CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DB_PATH = PROJECT_ROOT / 'data' / 'wallet_profiles.db'
THREAT_FILE_PATH = PROJECT_ROOT / 'data' / 'dark_web_wallets.txt'

# --- MODEL SELECTION ---
FORCED_MODEL_TIMESTAMP = None
if len(sys.argv) > 1:
    if not sys.argv[1].startswith('-'):
        FORCED_MODEL_TIMESTAMP = sys.argv[1]

print(f"[agents.py] Project root: {PROJECT_ROOT}")
print(f"[agents.py] DB path: {DB_PATH} (exists: {DB_PATH.exists()})")
if FORCED_MODEL_TIMESTAMP:
    print(f"[agents.py] Model: {FORCED_MODEL_TIMESTAMP}")
else:
    print(f"[agents.py] Model: 'latest'")

# --- AGENT 3 IMPORT ---
try:
    from app.agent_3_network import agent_3_network
    print("[agents.py] ✓ Agent 3 loaded")
except ImportError:
    try:
        from agent_3_network import agent_3_network
        print("[agents.py] ✓ Agent 3 loaded")
    except ImportError:
        print("[agents.py] ⚠️  Agent 3 not found")
        def agent_3_network(transaction: Dict) -> Dict:
            transaction['agent_3_score'] = 0
            transaction['agent_3_reasons'] = ["Agent 3 skipped"]
            return transaction

# --- GLOBAL CACHE ---
THREAT_LIST_CACHE = None
THREAT_LIST_LAST_LOADED = None
KNOWN_MIXER_ADDRESSES = set()  # Will be loaded from threat list
RECENT_SMALL_TRANSFERS = {}
TRANSFER_WINDOW = timedelta(minutes=10)
TRANSFER_THRESHOLD = 5
SMALL_TRANSFER_VALUE = 0.05


def load_dark_web_wallets(force_reload: bool = False) -> set:
    """Load threat list with caching"""
    global THREAT_LIST_CACHE, THREAT_LIST_LAST_LOADED, KNOWN_MIXER_ADDRESSES
    now = datetime.now()

    if THREAT_LIST_CACHE is None or force_reload or \
       (THREAT_LIST_LAST_LOADED and (now - THREAT_LIST_LAST_LOADED > timedelta(minutes=5))):

        if not THREAT_FILE_PATH.exists():
            print(f"Warning: {THREAT_FILE_PATH} not found.")
            THREAT_LIST_CACHE = set()
            KNOWN_MIXER_ADDRESSES = set()
        else:
            try:
                with open(THREAT_FILE_PATH, 'r') as f:
                    wallets = set(line.strip().lower() for line in f if line.strip())
                THREAT_LIST_CACHE = wallets
                # Mixers are those starting with '0xmix' or similar pattern
                KNOWN_MIXER_ADDRESSES = {w for w in wallets if 'mix' in w[:10].lower()}
                print(f"Loaded threat list: {len(THREAT_LIST_CACHE)} addresses ({len(KNOWN_MIXER_ADDRESSES)} mixers)")
            except Exception as e:
                print(f"Error loading {THREAT_FILE_PATH}: {e}")
                THREAT_LIST_CACHE = set()
                KNOWN_MIXER_ADDRESSES = set()
        THREAT_LIST_LAST_LOADED = now
    return THREAT_LIST_CACHE or set()


def agent_1_monitor(transaction: Dict) -> Dict:
    """Agent 1: Threat Intelligence & Heuristics"""
    score = 0
    reasons = []
    dark_web_wallets = load_dark_web_wallets()
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value_eth = transaction.get('value_eth', 0.0)
    timestamp_str = transaction.get('timestamp')

    # Known Threat List Check
    if from_addr in dark_web_wallets:
        score += 40
        reasons.append(f"From-Wallet on Threat List")

    if to_addr in dark_web_wallets:
        if to_addr in KNOWN_MIXER_ADDRESSES:
            score += 50
            reasons.append(f"To-Wallet is a known Mixer")
        else:
            score += 40
            reasons.append(f"To-Wallet on Threat List")

    # High Value Check
    if value_eth > 100:
        score += 20
        reasons.append(f"Very high-value: {value_eth:.4f} ETH")
    elif value_eth > 50:
        score += 10
        reasons.append(f"High-value: {value_eth:.4f} ETH")

    # Small Repetitive Transfers
    if value_eth > 0 and value_eth < SMALL_TRANSFER_VALUE and from_addr:
        now = datetime.now(UTC)
        if from_addr not in RECENT_SMALL_TRANSFERS:
            RECENT_SMALL_TRANSFERS[from_addr] = []
        
        RECENT_SMALL_TRANSFERS[from_addr] = [
            ts for ts in RECENT_SMALL_TRANSFERS[from_addr] if now - ts <= TRANSFER_WINDOW
        ]
        
        try:
            current_ts = datetime.fromisoformat(timestamp_str.replace('Z', '')).replace(tzinfo=UTC)
        except (ValueError, TypeError):
            current_ts = now
        
        RECENT_SMALL_TRANSFERS[from_addr].append(current_ts)
        
        count = len(RECENT_SMALL_TRANSFERS[from_addr])
        if count >= TRANSFER_THRESHOLD:
            score += 15
            reasons.append(f"High frequency of small transfers ({count})")

    # High Gas Price Check
    gas_price = transaction.get('gas_price', 0.0)
    if gas_price > 200:
        score += 10
        reasons.append(f"Very high gas: {gas_price:.2f} Gwei")

    # Self-Transfer
    if from_addr and to_addr and from_addr == to_addr and value_eth > 0:
        score += 5
        reasons.append("Self-transfer")

    transaction['agent_1_score'] = min(score, 100)
    transaction['agent_1_reasons'] = reasons
    return transaction


def get_wallet_profile_from_db(conn, wallet_address):
    """Query wallet profile from database"""
    if conn is None:
        return {
            'avg_sent': 0.001, 'tx_count_sent': 0, 'unique_recipients': 0,
            'avg_time_between_tx': 86400.0, 'min_time_between_tx': 86400.0,
            'tx_count_received': 0, 'total_received': 0.0, 'unique_senders': 0,
            'values_sent': [], 'std_sent': 0.0,
            'values_received': [], 'std_received': 0.0,
            'recipients_history': {}
        }
        
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            avg_sent_value, tx_count_sent, unique_recipients, 
            avg_time_between_tx, min_time_between_tx,
            tx_count_received, total_received, unique_senders
        FROM wallet_profiles 
        WHERE wallet_address = ?
    """, (wallet_address,))
    
    result = cursor.fetchone()
    if result:
        # Basic profile from DB
        profile = {
            'avg_sent': float(result[0]) if result[0] else 0.001,
            'tx_count_sent': int(result[1]) if result[1] else 0,
            'unique_recipients': int(result[2]) if result[2] else 0,
            'avg_time_between_tx': float(result[3]) if result[3] else 86400.0,
            'min_time_between_tx': float(result[4]) if result[4] else 86400.0,
            'tx_count_received': int(result[5]) if result[5] else 0,
            'total_received': float(result[6]) if result[6] else 0.0,
            'unique_senders': int(result[7]) if result[7] else 0,
        }
        # Add estimated std (we don't store historical values)
        profile['std_sent'] = profile['avg_sent'] * 0.3  # Rough estimate
        profile['std_received'] = profile.get('total_received', 0) * 0.2
        profile['values_sent'] = []
        profile['values_received'] = []
        profile['recipients_history'] = {}
        return profile
    else:
        return {
            'avg_sent': 0.001, 'tx_count_sent': 0, 'unique_recipients': 0,
            'avg_time_between_tx': 86400.0, 'min_time_between_tx': 86400.0,
            'tx_count_received': 0, 'total_received': 0.0, 'unique_senders': 0,
            'std_sent': 0.0, 'std_received': 0.0,
            'values_sent': [], 'values_received': [], 'recipients_history': {}
        }


def engineer_features_for_prediction(transaction: Dict) -> np.ndarray:
    """
    V4: Engineer 35 features (NO THREAT LIST) matching training
    """
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value = transaction.get('value_eth', 0.0)
    gas = transaction.get('gas_price', 0.0)

    feat = []

    # Get profiles from DB
    sender_profile = {}
    receiver_profile = {}
    
    if DB_PATH.exists():
        try:
            with sqlite3.connect(str(DB_PATH)) as conn:
                sender_profile = get_wallet_profile_from_db(conn, from_addr)
                receiver_profile = get_wallet_profile_from_db(conn, to_addr)
        except Exception as e:
            print(f"DB error: {e}")
            sender_profile = get_wallet_profile_from_db(None, None)
            receiver_profile = get_wallet_profile_from_db(None, None)
    else:
        sender_profile = get_wallet_profile_from_db(None, None)
        receiver_profile = get_wallet_profile_from_db(None, None)

    # ===== GROUP 1: BASIC (0-5) =====
    feat.extend([
        value,                          # 0
        gas,                            # 1
        value * gas,                    # 2
        np.log1p(value),                # 3
        np.log1p(gas),                  # 4
        1 if value > 0 else 0,          # 5
    ])

    # ===== GROUP 2: SENDER BEHAVIORAL (6-14) =====
    sender_tx_count = sender_profile['tx_count_sent']
    sender_avg = sender_profile['avg_sent']
    sender_std = sender_profile['std_sent']
    sender_unique_recipients = sender_profile['unique_recipients']
    
    feat.extend([
        sender_tx_count,                                    # 6
        np.log1p(sender_tx_count),                         # 7
        sender_avg,                                         # 8
        sender_std,                                         # 9
        value / max(sender_avg, 0.001),                    # 10
        abs(value - sender_avg) / max(sender_std, 0.001),  # 11
        sender_unique_recipients,                           # 12
        sender_unique_recipients / max(1, sender_tx_count), # 13
        1 if sender_tx_count < 5 else 0,                   # 14
    ])

    # ===== GROUP 3: RECEIVER BEHAVIORAL (15-21) =====
    receiver_tx_count = receiver_profile['tx_count_received']
    receiver_avg_received = receiver_profile['total_received'] / max(1, receiver_tx_count) if receiver_tx_count > 0 else 0.0
    receiver_unique_senders = receiver_profile['unique_senders']
    
    feat.extend([
        receiver_tx_count,                                  # 15
        receiver_avg_received,                              # 16
        receiver_unique_senders,                            # 17
        receiver_unique_senders / max(1, receiver_tx_count), # 18
        1 if receiver_tx_count < 5 else 0,                  # 19
        1 if receiver_unique_senders > 50 else 0,           # 20
        1 if sender_unique_recipients > 50 else 0,          # 21
    ])

    # ===== GROUP 4: TEMPORAL (22-27) =====
    try:
        ts = datetime.fromisoformat(transaction.get('timestamp', '').replace('Z', ''))
        hour, weekday = ts.hour, ts.weekday()
    except:
        hour, weekday = 12, 3
    
    feat.extend([
        hour / 24.0,                                        # 22
        weekday / 6.0,                                      # 23
        1 if 2 <= hour <= 5 else 0,                        # 24
        1 if weekday >= 5 else 0,                           # 25
        np.log1p(sender_profile['avg_time_between_tx']),   # 26
        np.log1p(sender_profile['min_time_between_tx']),   # 27
    ])

    # ===== GROUP 5: RELATIONSHIP (28-31) =====
    # We don't have recipients_history in real-time, so default to 0
    sender_to_receiver_history = 0
    
    feat.extend([
        sender_to_receiver_history,                         # 28
        1 if sender_to_receiver_history > 0 else 0,        # 29
        1 if from_addr == to_addr else 0,                  # 30
        len(set(from_addr) & set(to_addr)) / 42.0,         # 31
    ])

    # ===== GROUP 6: ANOMALY (32-34) =====
    feat.extend([
        1 if value > 100 else 0,                            # 32
        1 if value < 0.01 else 0,                           # 33
        gas / max(value, 0.001),                            # 34
    ])

    if len(feat) != 35:
        print(f"ERROR: Generated {len(feat)} features, expected 35!")
        
    return np.array(feat).reshape(1, -1)


def agent_2_behavioral(transaction: Dict, model_timestamp: str = None) -> Dict:
    """Agent 2: ML Behavioral Analysis (using 35 features)"""
    score = 0
    reasons = []

    # Load Model
    effective_timestamp = model_timestamp or FORCED_MODEL_TIMESTAMP 

    if effective_timestamp:
        model_path = PROJECT_ROOT / 'models' / f'fraud_model_{effective_timestamp}.pkl'
        scaler_path = PROJECT_ROOT / 'models' / f'scaler_{effective_timestamp}.pkl'
        metadata_path = PROJECT_ROOT / 'models' / f'model_metadata_{effective_timestamp}.json'
    else:
        try:
            with open(PROJECT_ROOT / 'models' / 'latest_model_timestamp.txt', 'r') as f:
                latest_timestamp = f.read().strip()
            effective_timestamp = latest_timestamp
            model_path = PROJECT_ROOT / 'models' / f'fraud_model_{latest_timestamp}.pkl'
            scaler_path = PROJECT_ROOT / 'models' / f'scaler_{latest_timestamp}.pkl'
            metadata_path = PROJECT_ROOT / 'models' / f'model_metadata_{latest_timestamp}.json'
        except FileNotFoundError:
            print("Warning: 'latest_model_timestamp.txt' not found.")
            model_path = PROJECT_ROOT / 'models' / 'fraud_model.pkl'
            scaler_path = PROJECT_ROOT / 'models' / 'scaler.pkl'
            metadata_path = PROJECT_ROOT / 'models' / 'model_metadata.json'
            effective_timestamp = "default"

    optimal_threshold = 0.5
    try:
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                optimal_threshold = metadata.get('optimal_threshold', 0.5)
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")

    # ML Prediction
    if model_path.exists() and scaler_path.exists():
        try:
            with open(model_path, 'rb') as f: 
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f: 
                scaler = pickle.load(f)

            features = engineer_features_for_prediction(transaction)
            features_scaled = scaler.transform(features)

            probability = model.predict_proba(features_scaled)[0]
            fraud_probability = probability[1] if len(probability) > 1 else 0.0
            prediction = 1 if fraud_probability >= optimal_threshold else 0
            
            model_id = f" (Model: {effective_timestamp})"
            
            if fraud_probability >= 0.90:
                score += 60
                reasons.append(f"ML: Very High Risk ({fraud_probability:.1%}){model_id}")
            elif fraud_probability >= 0.75:
                score += 40
                reasons.append(f"ML: High Risk ({fraud_probability:.1%}){model_id}")
            elif fraud_probability >= 0.60:
                score += 25
                reasons.append(f"ML: Moderate Risk ({fraud_probability:.1%}){model_id}")
            elif fraud_probability >= optimal_threshold:
                score += 10
                reasons.append(f"ML: Elevated Risk ({fraud_probability:.1%}){model_id}")

            transaction['ml_fraud_probability'] = round(fraud_probability, 4)
            transaction['ml_prediction'] = prediction

        except Exception as e:
            print(f"Warning: ML model error: {e}")
            transaction['ml_fraud_probability'] = -1.0
            transaction['ml_prediction'] = -1
    else:
        print(f"Warning: Model not found at {model_path}")
        transaction['ml_fraud_probability'] = -1.0
        transaction['ml_prediction'] = -1

    transaction['agent_2_score'] = min(score, 100)
    transaction['agent_2_reasons'] = reasons
    return transaction


def agent_4_decision(transaction: Dict) -> Dict:
    """Agent 4: Decision Aggregator"""
    agent_1_score = transaction.get('agent_1_score', 0)
    agent_2_score = transaction.get('agent_2_score', 0)
    agent_3_score = transaction.get('agent_3_score', 0)

    final_score = (agent_1_score * 0.8) + (agent_2_score * 1.2) + (agent_3_score * 0.5)
    
    all_reasons = []
    all_reasons.extend(transaction.get('agent_1_reasons', []))
    all_reasons.extend(transaction.get('agent_2_reasons', []))
    all_reasons.extend(transaction.get('agent_3_reasons', []))
    
    final_score = round(min(final_score, 150.0), 1)

    DENY_THRESHOLD = 75
    REVIEW_THRESHOLD = 45

    if final_score >= DENY_THRESHOLD:
        status = "DENY"
    elif final_score >= REVIEW_THRESHOLD:
        status = "FLAG_FOR_REVIEW"
    else:
        status = "APPROVE"

    transaction['final_score'] = final_score
    transaction['final_status'] = status
    transaction['reasons'] = " | ".join(all_reasons) if all_reasons else "No flags detected"
    return transaction


def process_transaction_pipeline(transaction: Dict, model_timestamp: str = None) -> Dict:
    """Pass transaction through the complete agent pipeline"""
    processed_tx = agent_1_monitor(transaction.copy())
    processed_tx = agent_2_behavioral(processed_tx, model_timestamp=model_timestamp)
    processed_tx = agent_3_network(processed_tx)
    processed_tx = agent_4_decision(processed_tx)
    if 'is_fraud' in transaction:
        processed_tx['is_fraud'] = transaction['is_fraud']
    return processed_tx


def get_system_thresholds():
    """Returns current system thresholds"""
    return {
        "agent_4_decision": {
            "deny_threshold": 75,
            "review_threshold": 45,
            "approve_threshold": 0
        },
        "agent_1_threat": {
            "threat_list_match": 40,
            "mixer_match": 50,
        },
        "agent_2_behavioral_ml": {
            "very_high_risk": 60,
            "high_risk": 40,
            "moderate_risk": 25,
        }
    }


def analyze_transaction(tx_hash: str, transaction: Dict, model_timestamp: str = None) -> Dict:
    """Detailed analysis for debugging"""
    processed = process_transaction_pipeline(transaction.copy(), model_timestamp=model_timestamp)
    analysis = {
        "tx_hash": tx_hash,
        "final_status": processed.get('final_status'),
        "final_score": processed.get('final_score'),
        "breakdown": {
            "agent_1": {
                "score": processed.get('agent_1_score', 0),
                "reasons": processed.get('agent_1_reasons', [])
            },
            "agent_2": {
                "score": processed.get('agent_2_score', 0),
                "reasons": processed.get('agent_2_reasons', []),
                "ml_probability": processed.get('ml_fraud_probability', 0)
            },
            "agent_3": {
                "score": processed.get('agent_3_score', 0),
                "reasons": processed.get('agent_3_reasons', [])
            }
        },
        "transaction_data": {
            "from": transaction.get('from_address'),
            "to": transaction.get('to_address'),
            "value": transaction.get('value_eth'),
            "gas": transaction.get('gas_price')
        }
    }
    return analysis


if __name__ == "__main__":
    """Test block"""
    print("=" * 60)
    print("Testing Agent Pipeline (V4.0)...")
    print("=" * 60)
    
    if not DB_PATH.exists() or not THREAT_FILE_PATH.exists():
        print("ERROR: Database or threat list not found.")
        print(f"Please run 'initialize_and_train.py' first!")
    else:
        test_tx_normal = {
            "tx_hash": "0xtest_normal_123",
            "from_address": "0xnorm000000000000000000000000000000000000",
            "to_address": "0xnorm000000000000000000000000000000000001",
            "value_eth": 2.5,
            "gas_price": 50.0,
            "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "is_fraud": 0
        }
        
        print("\n--- Testing NORMAL Transaction ---")
        result_normal = process_transaction_pipeline(test_tx_normal)
        print(f"Status: {result_normal['final_status']}")
        print(f"Score: {result_normal['final_score']}")
        print(f"Reasons: {result_normal['reasons']}")
        
        print("=" * 60)