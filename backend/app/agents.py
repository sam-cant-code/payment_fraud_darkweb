"""
Agent Logic Module for Blockchain Fraud Detection MVP
ROBUST VERSION with better path resolution
"""

import pickle
import sqlite3
from typing import Dict
import os
from datetime import datetime, timedelta, UTC
import numpy as np
from pathlib import Path

# --- ROBUST PATH RESOLUTION ---
def get_backend_dir():
    """Get the backend directory path reliably"""
    # Try multiple methods to find backend directory
    
    # Method 1: From this file's location
    current_file = Path(__file__).resolve()
    
    # If we're in backend/app/agents.py, go up to backend/
    if current_file.parent.name == 'app':
        return current_file.parent.parent
    
    # If we're directly in backend/
    if current_file.parent.name == 'backend':
        return current_file.parent
    
    # Method 2: From current working directory
    cwd = Path.cwd()
    
    # If current dir is backend/
    if cwd.name == 'backend':
        return cwd
    
    # If backend/ is a subdirectory
    backend_path = cwd / 'backend'
    if backend_path.exists():
        return backend_path
    
    # If we're in a subdirectory of backend
    if (cwd / 'app').exists() or (cwd / 'scripts').exists():
        return cwd
    
    # Method 3: Go up until we find backend/
    check_path = cwd
    for _ in range(5):  # Check up to 5 levels up
        if check_path.name == 'backend' or (check_path / 'app').exists():
            return check_path
        check_path = check_path.parent
    
    # Fallback: Use current working directory
    print(f"Warning: Could not reliably determine backend directory. Using: {cwd}")
    return cwd

BACKEND_DIR = get_backend_dir()
MODEL_PATH = BACKEND_DIR / 'models' / 'fraud_model.pkl'
SCALER_PATH = BACKEND_DIR / 'models' / 'scaler.pkl'
DB_PATH = BACKEND_DIR / 'data' / 'wallet_profiles.db'
THREAT_FILE_PATH = BACKEND_DIR / 'data' / 'dark_web_wallets.txt'

# Print paths on module load for debugging
print(f"[agents.py] Backend directory: {BACKEND_DIR}")
print(f"[agents.py] Model path: {MODEL_PATH} (exists: {MODEL_PATH.exists()})")
print(f"[agents.py] Scaler path: {SCALER_PATH} (exists: {SCALER_PATH.exists()})")

# Import the neutralized network agent
try:
    from agent_3_network import agent_3_network
except ImportError:
    print("Warning: agent_3_network.py not found. Agent 3 will be skipped.")
    def agent_3_network(transaction: Dict) -> Dict:
        transaction['agent_3_score'] = 0
        transaction['agent_3_reasons'] = ["Agent 3 skipped (module not found)"]
        return transaction

# --- Global Cache ---
THREAT_LIST_CACHE = None
THREAT_LIST_LAST_LOADED = None

KNOWN_MIXER_ADDRESSES = {f"0xmix{i:038x}" for i in range(2)}

RECENT_SMALL_TRANSFERS = {}
TRANSFER_WINDOW = timedelta(minutes=10)
TRANSFER_THRESHOLD = 5
SMALL_TRANSFER_VALUE = 0.01


def load_dark_web_wallets(force_reload: bool = False) -> set:
    """Load the list of known malicious wallet addresses with caching"""
    global THREAT_LIST_CACHE, THREAT_LIST_LAST_LOADED
    now = datetime.now()
    
    if THREAT_LIST_CACHE is None or force_reload or \
       (THREAT_LIST_LAST_LOADED and (now - THREAT_LIST_LAST_LOADED > timedelta(minutes=5))):
        
        if not THREAT_FILE_PATH.exists():
            print(f"Warning: {THREAT_FILE_PATH} not found. Threat list is empty.")
            THREAT_LIST_CACHE = set()
        else:
            try:
                with open(THREAT_FILE_PATH, 'r') as f:
                    wallets = set(line.strip().lower() for line in f if line.strip())
                THREAT_LIST_CACHE = wallets
                print(f"Loaded threat list: {len(THREAT_LIST_CACHE)} addresses.")
            except Exception as e:
                print(f"Error loading {THREAT_FILE_PATH}: {e}")
                THREAT_LIST_CACHE = set()
        THREAT_LIST_LAST_LOADED = now
    
    return THREAT_LIST_CACHE or set()


def agent_1_monitor(transaction: Dict) -> Dict:
    """Agent 1: Threat Intelligence Engine"""
    score = 0
    reasons = []
    dark_web_wallets = load_dark_web_wallets()
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value_eth = transaction.get('value_eth', 0.0)
    timestamp_str = transaction.get('timestamp')

    # Known Threat List Check
    if from_addr in dark_web_wallets:
        score += 60
        reasons.append(f"From-Wallet ({from_addr[:10]}...) on Threat List")
    if to_addr in dark_web_wallets:
        if to_addr in KNOWN_MIXER_ADDRESSES:
            score += 70
            reasons.append(f"To-Wallet ({to_addr[:10]}...) is a known Mixer")
        else:
            score += 60
            reasons.append(f"To-Wallet ({to_addr[:10]}...) on Threat List")

    # High Value Check
    if value_eth > 100:
        score += 40
        reasons.append(f"Very high-value transaction: {value_eth:.4f} ETH")
    elif value_eth > 50:
        score += 25
        reasons.append(f"High-value transaction: {value_eth:.4f} ETH")
    elif value_eth > 30:
        score += 15
        reasons.append(f"Elevated-value transaction: {value_eth:.4f} ETH")

    # Zero Value Check
    if value_eth == 0.0 and transaction.get('gas_price', 0) > 0:
        score += 3
        reasons.append("Zero ETH value transaction")

    # Small Repetitive Transfers
    if value_eth > 0 and value_eth < SMALL_TRANSFER_VALUE and from_addr:
        now = datetime.now(UTC)
        if from_addr in RECENT_SMALL_TRANSFERS:
            RECENT_SMALL_TRANSFERS[from_addr] = [
                ts for ts in RECENT_SMALL_TRANSFERS[from_addr] if now - ts <= TRANSFER_WINDOW
            ]
        else:
            RECENT_SMALL_TRANSFERS[from_addr] = []

        current_ts = None
        if timestamp_str:
            try:
                current_ts = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=UTC)
            except (ValueError, TypeError):
                current_ts = now

        if current_ts:
            RECENT_SMALL_TRANSFERS[from_addr].append(current_ts)

        count = len(RECENT_SMALL_TRANSFERS[from_addr])
        if count >= TRANSFER_THRESHOLD:
            score += 20
            reasons.append(f"High frequency of small transfers ({count})")

    # High Gas Price Check
    gas_price = transaction.get('gas_price', 0.0)
    if gas_price > 200:
        score += 15
        reasons.append(f"Unusually high gas price: {gas_price:.2f} Gwei")

    transaction['agent_1_score'] = score
    transaction['agent_1_reasons'] = reasons
    return transaction


def engineer_features_for_prediction(transaction: Dict) -> np.ndarray:
    """Engineer features for prediction"""
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value = transaction.get('value_eth', 0.0)
    gas = transaction.get('gas_price', 0.0)
    
    threat_set = set(w.lower() for w in load_dark_web_wallets())
    
    features = [value, gas, value * gas]
    features.append(1 if from_addr in threat_set else 0)
    features.append(1 if to_addr in threat_set else 0)
    
    sender_tx_count = 0
    sender_avg_value = 0.1
    
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT avg_transaction_value, transaction_count FROM wallet_profiles WHERE wallet_address = ?",
                (from_addr,)
            )
            result = cursor.fetchone()
            if result:
                sender_avg_value = float(result[0]) if result[0] else 0.1
                sender_tx_count = int(result[1]) if result[1] else 0
            conn.close()
        except Exception as e:
            print(f"DB error: {e}")
    
    features.append(sender_tx_count)
    features.append(sender_avg_value)
    features.append(value / max(sender_avg_value, 0.001))
    
    try:
        ts = datetime.fromisoformat(transaction.get('timestamp', '').replace('Z', ''))
        features.append(ts.hour)
        features.append(ts.weekday())
    except:
        features.append(12)
        features.append(3)
    
    features.append(86400)
    features.append(1 if value > 100 else 0)
    features.append(1 if value > 50 else 0)
    features.append(1 if gas > 150 else 0)
    features.append(gas / max(value, 0.001))
    
    return np.array(features).reshape(1, -1)


def agent_2_behavioral(transaction: Dict) -> Dict:
    """Agent 2: Behavioral Profiler with ML"""
    score = 0
    reasons = []

    # ML Model Prediction
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            
            features = engineer_features_for_prediction(transaction)
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            fraud_probability = probability[1] if len(probability) > 1 else 0.0
            
            ml_risk_score = int(fraud_probability * 60)
            
            if prediction == 1 or fraud_probability > 0.3:
                score += ml_risk_score
                reasons.append(f"ML Model: High fraud probability ({fraud_probability:.2%})")
            elif fraud_probability > 0.15:
                score += int(ml_risk_score * 0.8)
                reasons.append(f"ML Model: Moderate fraud probability ({fraud_probability:.2%})")
                
        except Exception as e:
            print(f"Warning: Error using ML model: {e}")
    else:
        print(f"Warning: Model files not found at {MODEL_PATH}")

    # Historical Profile Check
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            from_addr = transaction.get('from_address', '')
            if from_addr:
                cursor.execute(
                    "SELECT avg_transaction_value FROM wallet_profiles WHERE wallet_address = ?",
                    (from_addr.lower(),)
                )
                result = cursor.fetchone()
                if result and result[0] is not None:
                    avg_value = result[0]
                    current_value = transaction.get('value_eth', 0.0)
                    if avg_value > 0.0001:
                        deviation_ratio = current_value / avg_value
                        if deviation_ratio > 10:
                            score += 35
                            reasons.append(f"Value {deviation_ratio:.1f}x higher than avg")
                        elif deviation_ratio > 5:
                            score += 20
                            reasons.append(f"Value {deviation_ratio:.1f}x higher than avg")
                        elif deviation_ratio > 3:
                            score += 12
                            reasons.append(f"Value {deviation_ratio:.1f}x higher than avg")
                    elif current_value > 1:
                        score += 15
                        reasons.append(f"Significant value from low-activity wallet")
            conn.close()
        except Exception as e:
            print(f"Warning: DB error: {e}")

    transaction['agent_2_score'] = score
    transaction['agent_2_reasons'] = reasons
    return transaction


def agent_4_decision(transaction: Dict) -> Dict:
    """Agent 4: Decision Aggregator"""
    agent_1_score = transaction.get('agent_1_score', 0)
    agent_2_score = transaction.get('agent_2_score', 0)
    agent_3_score = transaction.get('agent_3_score', 0)

    final_score = (agent_1_score * 1.2) + (agent_2_score * 1.5) + (agent_3_score * 0.5)

    all_reasons = []
    all_reasons.extend(transaction.get('agent_1_reasons', []))
    all_reasons.extend(transaction.get('agent_2_reasons', []))
    all_reasons.extend(transaction.get('agent_3_reasons', []))

    final_score = round(min(final_score, 150.0), 1)

    DENY_THRESHOLD = 40
    REVIEW_THRESHOLD = 20

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


def process_transaction_pipeline(transaction: Dict) -> Dict:
    """Pass transaction through the complete agent pipeline"""
    is_fraud_label = transaction.get('is_fraud')

    processed_tx = agent_1_monitor(transaction.copy())
    processed_tx = agent_2_behavioral(processed_tx)
    processed_tx = agent_3_network(processed_tx)
    processed_tx = agent_4_decision(processed_tx)

    if is_fraud_label is not None:
        processed_tx['is_fraud'] = is_fraud_label

    return processed_tx