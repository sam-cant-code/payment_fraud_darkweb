"""
Agent Logic Module - CALIBRATED VERSION
Enhanced scoring with proper threat detection and balanced weights
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

# Model selection from command-line
FORCED_MODEL_TIMESTAMP = None
if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
    FORCED_MODEL_TIMESTAMP = sys.argv[1]

print(f"[agents.py] Model: {FORCED_MODEL_TIMESTAMP or 'latest'}")

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
KNOWN_MIXER_ADDRESSES = {f"0xmix{i:038x}" for i in range(10)}
RECENT_SMALL_TRANSFERS = {}
TRANSFER_WINDOW = timedelta(minutes=10)
TRANSFER_THRESHOLD = 5
SMALL_TRANSFER_VALUE = 0.05


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
    """
    Agent 1: Threat Intelligence & Heuristics (CALIBRATED)
    Max Score: ~100 points
    """
    score = 0
    reasons = []
    dark_web_wallets = load_dark_web_wallets()
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value_eth = transaction.get('value_eth', 0.0)
    timestamp_str = transaction.get('timestamp')

    # === CRITICAL THREATS (50-60 points) ===
    
    # Known Malicious Sender
    if from_addr in dark_web_wallets:
        score += 50
        reasons.append(f"CRITICAL: Sender on threat list ({from_addr[:10]}...)")

    # Known Malicious Receiver
    if to_addr in dark_web_wallets:
        if to_addr in KNOWN_MIXER_ADDRESSES:
            score += 60
            reasons.append(f"CRITICAL: Transaction to known mixer ({to_addr[:10]}...)")
        else:
            score += 50
            reasons.append(f"CRITICAL: Recipient on threat list ({to_addr[:10]}...)")

    # === HIGH-RISK PATTERNS (20-35 points) ===
    
    # Very High Value Transaction
    if value_eth > 100:
        score += 30
        reasons.append(f"Very high-value transaction: {value_eth:.2f} ETH")
    elif value_eth > 50:
        score += 15
        reasons.append(f"High-value transaction: {value_eth:.2f} ETH")

    # Small Repetitive Transfers (Structuring)
    if value_eth > 0 and value_eth < SMALL_TRANSFER_VALUE and from_addr:
        now = datetime.now(UTC)
        if from_addr not in RECENT_SMALL_TRANSFERS:
            RECENT_SMALL_TRANSFERS[from_addr] = []
        
        # Clean old transfers
        RECENT_SMALL_TRANSFERS[from_addr] = [
            ts for ts in RECENT_SMALL_TRANSFERS[from_addr] 
            if now - ts <= TRANSFER_WINDOW
        ]
        
        try:
            current_ts = datetime.fromisoformat(timestamp_str.replace('Z', '')).replace(tzinfo=UTC)
        except (ValueError, TypeError):
            current_ts = now
        
        RECENT_SMALL_TRANSFERS[from_addr].append(current_ts)
        
        count = len(RECENT_SMALL_TRANSFERS[from_addr])
        if count >= TRANSFER_THRESHOLD:
            score += 25
            reasons.append(f"Structuring pattern: {count} small transfers in {TRANSFER_WINDOW.seconds//60} min")

    # === MEDIUM-RISK PATTERNS (10-20 points) ===
    
    # Very High Gas Price (Urgency indicator)
    gas_price = transaction.get('gas_price', 0.0)
    if gas_price > 200:
        score += 15
        reasons.append(f"Very high gas price: {gas_price:.2f} Gwei (urgency indicator)")
    elif gas_price > 150:
        score += 8
        reasons.append(f"High gas price: {gas_price:.2f} Gwei")

    # Suspiciously Low Gas (Potential test transaction)
    if gas_price > 0 and gas_price < 10:
        score += 5
        reasons.append(f"Unusually low gas: {gas_price:.2f} Gwei")

    # === LOW-RISK INDICATORS (5-10 points) ===
    
    # Self-Transfer
    if from_addr and to_addr and from_addr == to_addr and value_eth > 0:
        score += 10
        reasons.append("Self-transfer detected (possible wallet consolidation)")

    # Round Number Transaction (Potential test)
    if value_eth > 0:
        # Check if it's a very round number (e.g., 1.0, 10.0, 100.0)
        if value_eth in [1.0, 5.0, 10.0, 50.0, 100.0]:
            score += 5
            reasons.append(f"Round number transaction: {value_eth} ETH")

    # === TEMPORAL ANOMALIES ===
    
    try:
        tx_time = datetime.fromisoformat(timestamp_str.replace('Z', ''))
        hour = tx_time.hour
        
        # Late night transactions (2 AM - 5 AM)
        if 2 <= hour <= 5:
            score += 8
            reasons.append(f"Transaction at unusual hour: {hour:02d}:00")
    except:
        pass

    # Cap at 100
    final_score = min(score, 100)
    
    transaction['agent_1_score'] = final_score
    transaction['agent_1_reasons'] = reasons if reasons else ["No immediate threats detected"]
    
    return transaction


def get_wallet_profile_from_db(conn, wallet_address):
    """Helper to query the V3 database schema."""
    if conn is None:
        return {
            'avg_sent': 0.001, 'tx_count_sent': 0, 'unique_recipients': 0,
            'avg_time_between_tx': 86400.0, 'min_time_between_tx': 86400.0,
            'tx_count_received': 0, 'total_received': 0.0, 'unique_senders': 0,
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
        return {
            'avg_sent': float(result[0]) if result[0] else 0.001,
            'tx_count_sent': int(result[1]) if result[1] else 0,
            'unique_recipients': int(result[2]) if result[2] else 0,
            'avg_time_between_tx': float(result[3]) if result[3] else 86400.0,
            'min_time_between_tx': float(result[4]) if result[4] else 86400.0,
            'tx_count_received': int(result[5]) if result[5] else 0,
            'total_received': float(result[6]) if result[6] else 0.0,
            'unique_senders': int(result[7]) if result[7] else 0,
        }
    else:
        return {
            'avg_sent': 0.001, 'tx_count_sent': 0, 'unique_recipients': 0,
            'avg_time_between_tx': 86400.0, 'min_time_between_tx': 86400.0,
            'tx_count_received': 0, 'total_received': 0.0, 'unique_senders': 0,
        }


def engineer_features_for_prediction(transaction: Dict) -> np.ndarray:
    """V3: Engineer features that EXACTLY match V3 training data."""
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value = transaction.get('value_eth', 0.0)
    gas = transaction.get('gas_price', 0.0)

    threat_set = load_dark_web_wallets()

    # Basic features (0-2)
    feat = [value, gas, value * gas]

    # Threat intelligence (3-4)
    feat.extend([
        1 if from_addr in threat_set else 0,
        1 if to_addr in threat_set else 0,
    ])

    # Profile features from DB (5-17)
    sender_profile = {}
    receiver_profile = {}
    
    if DB_PATH.exists():
        try:
            with sqlite3.connect(str(DB_PATH)) as conn:
                sender_profile = get_wallet_profile_from_db(conn, from_addr)
                receiver_profile = get_wallet_profile_from_db(conn, to_addr)
        except Exception as e:
            print(f"DB error in feature engineering: {e}")
            sender_profile = get_wallet_profile_from_db(None, None)
            receiver_profile = get_wallet_profile_from_db(None, None)
    else:
        sender_profile = get_wallet_profile_from_db(None, None)
        receiver_profile = get_wallet_profile_from_db(None, None)

    # Sender profile (5-9)
    sender_avg_sent = sender_profile['avg_sent']
    sender_tx_count = sender_profile['tx_count_sent']
    sender_unique_recipients = sender_profile['unique_recipients']
    
    feat.extend([
        sender_tx_count,
        sender_avg_sent,
        value / max(sender_avg_sent, 0.001),
        sender_unique_recipients,
        sender_unique_recipients / max(1, sender_tx_count)
    ])
    
    # Receiver profile (10-12)
    feat.extend([
        receiver_profile['tx_count_received'],
        receiver_profile['total_received'],
        receiver_profile['unique_senders']
    ])
    
    # Temporal features (13-15)
    try:
        ts = datetime.fromisoformat(transaction.get('timestamp', '').replace('Z', ''))
        hour, weekday = ts.hour, ts.weekday()
        odd_hours = 1 if 2 <= hour <= 5 else 0
    except:
        hour, weekday, odd_hours = 12, 3, 0
    feat.extend([hour, weekday, odd_hours])

    # Sender temporal features (16-17)
    feat.extend([
        sender_profile['avg_time_between_tx'],
        sender_profile['min_time_between_tx']
    ])
    
    # Pattern features (18-25)
    feat.extend([
        1 if value > 100 else 0,
        1 if value > 50 else 0,
        1 if value < 0.1 else 0,
        1 if gas > 150 else 0,
        1 if gas < 30 else 0,
        gas / max(value, 0.001),
        1 if from_addr == to_addr else 0,
        1 if (value > 0 and (value * 100) % 100 < 0.01) else 0,
    ])

    if len(feat) != 26:
        print(f"ERROR: Feature engineering created {len(feat)} features, expected 26.")
        
    return np.array(feat).reshape(1, -1)


def agent_2_behavioral(transaction: Dict, model_timestamp: str = None) -> Dict:
    """
    Agent 2: Behavioral Profiler (ML Model) (CALIBRATED)
    Max Score: ~70 points
    """
    score = 0
    reasons = []

    # Load model
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

    # ML Model Prediction
    if model_path.exists() and scaler_path.exists():
        try:
            with open(model_path, 'rb') as f: model = pickle.load(f)
            with open(scaler_path, 'rb') as f: scaler = pickle.load(f)

            features = engineer_features_for_prediction(transaction)
            features_scaled = scaler.transform(features)

            probability = model.predict_proba(features_scaled)[0]
            fraud_probability = probability[1] if len(probability) > 1 else 0.0
            prediction = 1 if fraud_probability >= optimal_threshold else 0
            
            model_id = f" (Model: {effective_timestamp})"
            
            # CALIBRATED SCORING
            if fraud_probability >= 0.90:
                score += 70
                reasons.append(f"ML: Critical fraud risk ({fraud_probability:.1%}){model_id}")
            elif fraud_probability >= 0.80:
                score += 55
                reasons.append(f"ML: Very high fraud risk ({fraud_probability:.1%}){model_id}")
            elif fraud_probability >= 0.70:
                score += 40
                reasons.append(f"ML: High fraud risk ({fraud_probability:.1%}){model_id}")
            elif fraud_probability >= 0.60:
                score += 25
                reasons.append(f"ML: Elevated fraud risk ({fraud_probability:.1%}){model_id}")
            elif fraud_probability >= optimal_threshold:
                score += 15
                reasons.append(f"ML: Moderate fraud risk ({fraud_probability:.1%}){model_id}")

            transaction['ml_fraud_probability'] = round(fraud_probability, 4)
            transaction['ml_prediction'] = prediction

        except Exception as e:
            print(f"Warning: Error using ML model: {e}")
            transaction['ml_fraud_probability'] = -1.0
            transaction['ml_prediction'] = -1
    else:
        transaction['ml_fraud_probability'] = -1.0
        transaction['ml_prediction'] = -1

    final_score = min(score, 70)
    transaction['agent_2_score'] = final_score
    transaction['agent_2_reasons'] = reasons if reasons else ["No behavioral anomalies detected"]
    
    return transaction


def agent_4_decision(transaction: Dict) -> Dict:
    """
    Agent 4: Decision Aggregator (RECALIBRATED)
    
    Scoring System:
    - Agent 1 (Threat Intel): 0-100, Weight: 1.0
    - Agent 2 (ML Behavioral): 0-70, Weight: 1.3
    - Agent 3 (Network): 0-100, Weight: 0.8
    
    Max theoretical score: (100 * 1.0) + (70 * 1.3) + (100 * 0.8) = 271
    But we cap at 150 for cleaner display
    
    Thresholds:
    - DENY: >= 80 (very high confidence fraud)
    - FLAG_FOR_REVIEW: >= 45 (suspicious, needs review)
    - APPROVE: < 45 (low risk)
    """
    agent_1_score = transaction.get('agent_1_score', 0)
    agent_2_score = transaction.get('agent_2_score', 0)
    agent_3_score = transaction.get('agent_3_score', 0)

    # Weighted score calculation
    final_score = (agent_1_score * 1.0) + (agent_2_score * 1.3) + (agent_3_score * 0.8)
    
    # Collect all reasons
    all_reasons = []
    all_reasons.extend(transaction.get('agent_1_reasons', []))
    all_reasons.extend(transaction.get('agent_2_reasons', []))
    all_reasons.extend(transaction.get('agent_3_reasons', []))
    
    # Cap at 150
    final_score = round(min(final_score, 150.0), 1)

    # Decision thresholds
    DENY_THRESHOLD = 80
    REVIEW_THRESHOLD = 45

    # Priority override: If Agent 1 detects critical threat, escalate
    if agent_1_score >= 50:  # Critical threat detected
        if final_score < REVIEW_THRESHOLD:
            final_score = REVIEW_THRESHOLD  # Ensure at least flagged
        all_reasons.insert(0, "⚠️ CRITICAL THREAT DETECTED - Escalated")

    # Determine status
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
            "deny_threshold": 80,
            "review_threshold": 45,
            "approve_threshold": 0
        },
        "agent_1_threat": {
            "critical_threat": 50,
            "high_value": 30,
            "structuring": 25,
            "high_gas": 15
        },
        "agent_2_behavioral_ml": {
            "critical_risk": 70,
            "very_high_risk": 55,
            "high_risk": 40,
            "elevated_risk": 25
        },
        "agent_3_network": {
            "circular_direct": 45,
            "circular_short": 35,
            "high_fanout": 35,
            "high_fanin": 35,
            "rapid_velocity": 30
        },
        "weights": {
            "agent_1": 1.0,
            "agent_2": 1.3,
            "agent_3": 0.8
        }
    }


def analyze_transaction(tx_hash: str, transaction: Dict, model_timestamp: str = None) -> Dict:
    """Detailed analysis function for debugging"""
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
    print("Testing Calibrated Agent Pipeline")
    print("=" * 60)
    
    if not DB_PATH.exists() or not THREAT_FILE_PATH.exists():
        print("ERROR: Database or threat list not found.")
        print("Please run 'initialize_and_train.py' first!")
    else:
        # Test normal transaction
        test_tx_normal = {
            "tx_hash": "0xtest_normal_123",
            "from_address": "0xnorm000000000000000000000000000000000000",
            "to_address": "0xnorm000000000000000000000000000000000001",
            "value_eth": 2.5,
            "gas_price": 50.0,
            "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "is_fraud": 0
        }
        
        # Test fraudulent transaction
        test_tx_fraud = {
            "tx_hash": "0xtest_fraud_456",
            "from_address": "0xbad0000000000000000000000000000000000000",
            "to_address": "0xnorm000000000000000000000000000000000002",
            "value_eth": 150.5,
            "gas_price": 250.0,
            "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "is_fraud": 1
        }

        print("\n--- Testing NORMAL Transaction ---")
        result_normal = process_transaction_pipeline(test_tx_normal)
        print(f"Final Status: {result_normal['final_status']}")
        print(f"Final Score: {result_normal['final_score']}")
        print(f"Agent Scores: A1={result_normal['agent_1_score']}, A2={result_normal['agent_2_score']}, A3={result_normal['agent_3_score']}")
        print(f"Reasons: {result_normal['reasons']}")
        
        print("\n--- Testing FRAUDULENT Transaction ---")
        result_fraud = process_transaction_pipeline(test_tx_fraud)
        print(f"Final Status: {result_fraud['final_status']}")
        print(f"Final Score: {result_fraud['final_score']}")
        print(f"Agent Scores: A1={result_fraud['agent_1_score']}, A2={result_fraud['agent_2_score']}, A3={result_fraud['agent_3_score']}")
        print(f"Reasons: {result_fraud['reasons']}")
        
        print("\n" + "=" * 60)
        print("Thresholds:")
        thresholds = get_system_thresholds()
        print(f"  DENY: >= {thresholds['agent_4_decision']['deny_threshold']}")
        print(f"  FLAG_FOR_REVIEW: >= {thresholds['agent_4_decision']['review_threshold']}")
        print(f"  Agent Weights: A1={thresholds['weights']['agent_1']}, A2={thresholds['weights']['agent_2']}, A3={thresholds['weights']['agent_3']}")
        print("=" * 60)