"""
Agent Logic Module for Blockchain Fraud Detection MVP
IMPROVED VERSION - Calibrated for better accuracy with reduced false positives
"""

import pickle
import sqlite3
import json
from typing import Dict, Optional
import os
from datetime import datetime, timedelta, UTC
import numpy as np
from pathlib import Path

# --- PATH CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent  # backend/app/
BACKEND_DIR = SCRIPT_DIR.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # project root

# *** MODIFICATION: These are now DEFAULTS ***
MODEL_PATH = PROJECT_ROOT / 'models' / 'fraud_model.pkl'
SCALER_PATH = PROJECT_ROOT / 'models' / 'scaler.pkl'
DB_PATH = PROJECT_ROOT / 'data' / 'wallet_profiles.db'
THREAT_FILE_PATH = PROJECT_ROOT / 'data' / 'dark_web_wallets.txt'

print(f"[agents.py] Project root: {PROJECT_ROOT}")
print(f"[agents.py] Default Model path: {MODEL_PATH} (exists: {MODEL_PATH.exists()})")

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

KNOWN_MIXER_ADDRESSES = {f"0xmix{i:038x}" for i in range(10)} # Match new script

RECENT_SMALL_TRANSFERS = {}
TRANSFER_WINDOW = timedelta(minutes=10)
TRANSFER_THRESHOLD = 5
SMALL_TRANSFER_VALUE = 0.05 # Match new script


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
    Agent 1: Threat Intelligence Engine
    IMPROVED: Calibrated scoring to reduce false positives
    """
    score = 0
    reasons = []
    dark_web_wallets = load_dark_web_wallets()
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value_eth = transaction.get('value_eth', 0.0)
    timestamp_str = transaction.get('timestamp')

    # =====================================================
    # Known Threat List Check - REDUCED SCORES
    # =====================================================
    if from_addr in dark_web_wallets:
        score += 50  # Was 60
        reasons.append(f"From-Wallet ({from_addr[:10]}...) on Threat List")

    if to_addr in dark_web_wallets:
        if to_addr in KNOWN_MIXER_ADDRESSES:
            score += 60  # Was 70
            reasons.append(f"To-Wallet ({to_addr[:10]}...) is a known Mixer")
        else:
            score += 50  # Was 60
            reasons.append(f"To-Wallet ({to_addr[:10]}...) on Threat List")

    # =====================================================
    # High Value Check - RECALIBRATED THRESHOLDS
    # =====================================================
    if value_eth > 150:
        # Extremely high value
        score += 35  # Was 40
        reasons.append(f"Extremely high-value transaction: {value_eth:.4f} ETH")
    elif value_eth > 100:
        # Very high value
        score += 28  # Was 40
        reasons.append(f"Very high-value transaction: {value_eth:.4f} ETH")
    elif value_eth > 70:
        # High value
        score += 20  # Was 25
        reasons.append(f"High-value transaction: {value_eth:.4f} ETH")
    elif value_eth > 50:
        # Elevated value
        score += 15  # Was 25
        reasons.append(f"Elevated-value transaction: {value_eth:.4f} ETH")
    elif value_eth > 30:
        # Moderate-high value
        score += 8  # Was 15
        reasons.append(f"Moderate-high value transaction: {value_eth:.4f} ETH")

    # =====================================================
    # Zero Value Check - REDUCED IMPACT
    # =====================================================
    if value_eth == 0.0 and transaction.get('gas_price', 0) > 0:
        score += 2  # Was 3
        reasons.append("Zero ETH value transaction")

    # =====================================================
    # Small Repetitive Transfers - SLIGHTLY REDUCED
    # =====================================================
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
                current_ts = datetime.fromisoformat(timestamp_str.replace('Z', '')).replace(tzinfo=UTC)
            except (ValueError, TypeError):
                current_ts = now

        if current_ts:
            RECENT_SMALL_TRANSFERS[from_addr].append(current_ts)

        count = len(RECENT_SMALL_TRANSFERS[from_addr])
        if count >= TRANSFER_THRESHOLD:
            score += 18  # Was 20
            reasons.append(f"High frequency of small transfers ({count})")

    # =====================================================
    # High Gas Price Check - RECALIBRATED
    # =====================================================
    gas_price = transaction.get('gas_price', 0.0)
    if gas_price > 300:
        # Extremely high gas
        score += 20  # Was 15
        reasons.append(f"Extremely high gas price: {gas_price:.2f} Gwei")
    elif gas_price > 200:
        # Very high gas
        score += 12  # Was 15
        reasons.append(f"Very high gas price: {gas_price:.2f} Gwei")
    elif gas_price > 150:
        # High gas
        score += 6
        reasons.append(f"High gas price: {gas_price:.2f} Gwei")

    # =====================================================
    # Suspicious Pattern: Self-Transfer
    # =====================================================
    if from_addr and to_addr and from_addr == to_addr and value_eth > 0:
        score += 8
        reasons.append("Self-transfer detected")

    transaction['agent_1_score'] = score
    transaction['agent_1_reasons'] = reasons
    return transaction


def engineer_features_for_prediction(transaction: Dict) -> np.ndarray:
    """
    Engineer features for ML prediction
    *** MODIFIED: This now creates the 26 features that the model is *actually* trained on. ***
    """
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value = transaction.get('value_eth', 0.0)
    gas = transaction.get('gas_price', 0.0)

    threat_set = load_dark_web_wallets()

    # 0, 1, 2: Basic features
    feat = [
        value,
        gas,
        value * gas,
    ]

    # 3, 4: Threat intelligence
    feat.extend([
        1 if from_addr in threat_set else 0,
        1 if to_addr in threat_set else 0,
    ])

    # --- Profile features from DB ---
    sender_tx_count = 0
    sender_avg_value = 0.001 # Use 0.001 to avoid div by zero
    receiver_tx_count = 0
    
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            # Get sender profile
            cursor.execute(
                "SELECT avg_transaction_value, transaction_count FROM wallet_profiles WHERE wallet_address = ?",
                (from_addr,)
            )
            sender_result = cursor.fetchone()
            if sender_result:
                sender_avg_value = float(sender_result[0]) if sender_result[0] and sender_result[0] > 0 else 0.001
                sender_tx_count = int(sender_result[1]) if sender_result[1] else 0
            
            # Get receiver profile
            cursor.execute(
                "SELECT transaction_count FROM wallet_profiles WHERE wallet_address = ?",
                (to_addr,)
            )
            receiver_result = cursor.fetchone()
            if receiver_result:
                receiver_tx_count = int(receiver_result[0]) if receiver_result[0] else 0
                
            conn.close()
        except Exception as e:
            print(f"DB error in feature engineering: {e}")
    
    # 5, 6, 7: Sender profile
    feat.extend([
        sender_tx_count,
        sender_avg_value,
        value / max(sender_avg_value, 0.001) # value_deviation
    ])
    
    # 8, 9: Sender network (HARDCODED - model is trained on 0)
    feat.extend([
        0, # 8: sender_unique_recipients_HARDCODED
        0  # 9: recipient_diversity_HARDCODED
    ])
    
    # 10, 11, 12: Receiver profile
    feat.extend([
        receiver_tx_count,                     # 10: receiver_tx_count
        0, # 11: receiver_total_received_HARDCODED
        0  # 12: receiver_unique_senders_HARDCODED
    ])
    
    # 13, 14, 15: Temporal features
    try:
        ts = datetime.fromisoformat(transaction.get('timestamp', '').replace('Z', ''))
        hour = ts.hour
        weekday = ts.weekday()
        odd_hours = 1 if 2 <= hour <= 5 else 0
    except:
        hour = 12
        weekday = 3
        odd_hours = 0
        
    feat.extend([
        hour,
        weekday,
        odd_hours
    ])

    # 16, 17: Velocity features (HARDCODED - model is trained on 86400)
    feat.extend([
        86400, # 16: avg_time_between_tx_HARDCODED
        86400  # 17: min_time_between_tx_HARDCODED
    ])
    
    # 18-25: Pattern features
    feat.extend([
        1 if value > 100 else 0,             # 18: very_high_value
        1 if value > 50 else 0,              # 19: high_value
        1 if value < 0.1 else 0,             # 20: micro_value (matches new script's 0.05)
        1 if gas > 150 else 0,               # 21: high_gas
        1 if gas < 30 else 0,                # 22: low_gas
        gas / max(value, 0.001),             # 23: gas_value_ratio
        1 if from_addr == to_addr else 0,     # 24: self_transfer
        1 if (value > 0 and (value * 100) % 100 < 0.01) else 0, # 25: round_number
    ])

    # Should be 26 features total
    if len(feat) != 26:
        print(f"FATAL ERROR: Feature engineering created {len(feat)} features, expected 26.")
        
    return np.array(feat).reshape(1, -1)


def agent_2_behavioral(transaction: Dict, model_timestamp: str = None) -> Dict:
    """
    Agent 2: Behavioral Profiler with Machine Learning
    IMPROVED: Calibrated ML scoring and better historical profiling
    *** MODIFIED: Accepts model_timestamp to load specific models ***
    """
    score = 0
    reasons = []

    # =====================================================
    # ML Model Prediction - CALIBRATED SCORING
    # =====================================================

    # *** MODIFICATION: Select model paths based on timestamp ***
    if model_timestamp:
        model_path = PROJECT_ROOT / 'models' / f'fraud_model_{model_timestamp}.pkl'
        scaler_path = PROJECT_ROOT / 'models' / f'scaler_{model_timestamp}.pkl'
        metadata_path = PROJECT_ROOT / 'models' / f'model_metadata_{model_timestamp}.json'
    else:
        # Fallback to latest
        try:
            with open(PROJECT_ROOT / 'models' / 'latest_model_timestamp.txt', 'r') as f:
                latest_timestamp = f.read().strip()
            # print(f"[agents.py] No timestamp provided, using latest: {latest_timestamp}")
            model_path = PROJECT_ROOT / 'models' / f'fraud_model_{latest_timestamp}.pkl'
            scaler_path = PROJECT_ROOT / 'models' / f'scaler_{latest_timestamp}.pkl'
            metadata_path = PROJECT_ROOT / 'models' / f'model_metadata_{latest_timestamp}.json'
        except FileNotFoundError:
            print("[agents.py] 'latest_model_timestamp.txt' not found. Using default paths.")
            model_path = MODEL_PATH  # Use default
            scaler_path = SCALER_PATH # Use default
            metadata_path = PROJECT_ROOT / 'models' / 'model_metadata.json'


    # --- Load Optimal Threshold from metadata ---
    optimal_threshold = 0.5 # Default
    try:
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                optimal_threshold = metadata.get('optimal_threshold', 0.5)
                # print(f"[agents.py] Loaded optimal threshold: {optimal_threshold}")
    except Exception as e:
        print(f"Warning: Could not load metadata {metadata_path}. Using default 0.5 threshold. Error: {e}")


    if model_path.exists() and scaler_path.exists():
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            features = engineer_features_for_prediction(transaction)
            features_scaled = scaler.transform(features)

            # prediction = model.predict(features_scaled)[0] # Don't use this, use threshold
            probability = model.predict_proba(features_scaled)[0]
            fraud_probability = probability[1] if len(probability) > 1 else 0.0

            # ***MODIFICATION: Use optimal threshold for decision***
            prediction = 1 if fraud_probability >= optimal_threshold else 0
            
            # Use probability for scoring
            if fraud_probability >= 0.80:
                ml_risk_score = 50
                score += ml_risk_score
                reasons.append(f"ML Model: Very high fraud probability ({fraud_probability:.2%})")

            elif fraud_probability >= 0.65:
                ml_risk_score = 40
                score += ml_risk_score
                reasons.append(f"ML Model: High fraud probability ({fraud_probability:.2%})")

            elif fraud_probability >= optimal_threshold: # Use optimal threshold
                ml_risk_score = 30
                score += ml_risk_score
                reasons.append(f"ML Model: Moderate-high fraud probability ({fraud_probability:.2%})")

            elif fraud_probability >= 0.25:
                ml_risk_score = 10
                score += ml_risk_score
                reasons.append(f"ML Model: Low-moderate fraud probability ({fraud_probability:.2%})")

            # Store for analysis
            transaction['ml_fraud_probability'] = round(fraud_probability, 4)
            transaction['ml_prediction'] = prediction # Store the 0/1 prediction

        except Exception as e:
            print(f"Warning: Error using ML model: {e}")
            transaction['ml_fraud_probability'] = -1.0
            transaction['ml_prediction'] = -1
    else:
        if model_timestamp:
            print(f"Warning: Model files not found for timestamp '{model_timestamp}' at {model_path}")
        else:
            print(f"Warning: Default model files not found at {model_path}")
        transaction['ml_fraud_probability'] = -1.0
        transaction['ml_prediction'] = -1


    # =====================================================
    # Historical Profile Check - REFINED SCORING
    # =====================================================
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            from_addr = transaction.get('from_address', '')

            if from_addr:
                cursor.execute(
                    "SELECT avg_transaction_value, transaction_count FROM wallet_profiles WHERE wallet_address = ?",
                    (from_addr.lower(),)
                )
                result = cursor.fetchone()

                if result and result[0] is not None:
                    avg_value = float(result[0])
                    tx_count = int(result[1]) if result[1] else 0
                    current_value = transaction.get('value_eth', 0.0)

                    if avg_value > 0.0001:
                        deviation_ratio = current_value / avg_value

                        # IMPROVED: More nuanced deviation bands
                        if deviation_ratio > 15:
                            score += 40
                            reasons.append(f"Value {deviation_ratio:.1f}x higher than avg (extreme deviation)")
                        elif deviation_ratio > 10:
                            score += 30
                            reasons.append(f"Value {deviation_ratio:.1f}x higher than avg (very high deviation)")
                        elif deviation_ratio > 7:
                            score += 22
                            reasons.append(f"Value {deviation_ratio:.1f}x higher than avg (high deviation)")
                        elif deviation_ratio > 5:
                            score += 15
                            reasons.append(f"Value {deviation_ratio:.1f}x higher than avg (moderate-high deviation)")
                        elif deviation_ratio > 3:
                            score += 8
                            reasons.append(f"Value {deviation_ratio:.1f}x higher than avg (moderate deviation)")

                        # Check low-activity + high value
                        if tx_count < 10 and current_value > 10:
                            score += 12
                            reasons.append(f"High value from low-activity wallet ({tx_count} txs)")

                    elif current_value > 1:
                        score += 18
                        reasons.append(f"Significant value from near-zero avg wallet")

                else:
                    # Unknown wallet with high value
                    current_value = transaction.get('value_eth', 0.0)
                    if current_value > 20:
                        score += 15
                        reasons.append(f"High-value transaction from unknown wallet")

            conn.close()
        except Exception as e:
            print(f"Warning: DB error: {e}")

    # =====================================================
    # Behavioral Pattern Analysis
    # =====================================================
    value = transaction.get('value_eth', 0.0)
    gas_price = transaction.get('gas_price', 0.0)

    # Round number transactions
    if value > 0:
        if (value > 0 and (value * 100) % 100 < 0.01):
            score += 3
            reasons.append(f"Round number transaction ({value} ETH)")

    # Low gas for high value
    if value > 10 and gas_price < 20:
        score += 5
        reasons.append(f"Unusually low gas ({gas_price:.1f} Gwei) for high value")

    # Cap score
    score = min(score, 100)

    transaction['agent_2_score'] = score
    transaction['agent_2_reasons'] = reasons
    return transaction


def agent_4_decision(transaction: Dict) -> Dict:
    """
    Agent 4: Decision Aggregator
    IMPROVED: Balanced weights and higher thresholds
    """
    agent_1_score = transaction.get('agent_1_score', 0)
    agent_2_score = transaction.get('agent_2_score', 0)
    agent_3_score = transaction.get('agent_3_score', 0)

    # IMPROVED: Balanced weighting (reduced ML dominance)
    final_score = (agent_1_score * 1.0) + (agent_2_score * 1.0) + (agent_3_score * 0.5)

    all_reasons = []
    all_reasons.extend(transaction.get('agent_1_reasons', []))
    all_reasons.extend(transaction.get('agent_2_reasons', []))
    all_reasons.extend(transaction.get('agent_3_reasons', []))

    final_score = round(min(final_score, 150.0), 1)

    # IMPROVED: Higher thresholds to reduce false positives
    DENY_THRESHOLD = 70      # Was 40
    REVIEW_THRESHOLD = 45    # Was 20

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
    """
    Pass transaction through the complete agent pipeline
    *** MODIFIED: Accepts and passes model_timestamp ***
    """
    is_fraud_label = transaction.get('is_fraud')

    processed_tx = agent_1_monitor(transaction.copy())
    # *** MODIFICATION: Pass timestamp to agent_2 ***
    processed_tx = agent_2_behavioral(processed_tx, model_timestamp=model_timestamp)
    processed_tx = agent_3_network(processed_tx)
    processed_tx = agent_4_decision(processed_tx)

    if is_fraud_label is not None:
        processed_tx['is_fraud'] = is_fraud_label

    return processed_tx


def get_system_thresholds():
    """
    Returns current system thresholds for documentation/UI
    """
    return {
        "agent_4_decision": {
            "deny_threshold": 70,
            "review_threshold": 45,
            "approve_threshold": 0
        },
        "agent_1_threat": {
            "threat_list_match": 50,
            "mixer_match": 60,
            "very_high_value": 28,
            "high_gas": 12
        },
        "agent_2_behavioral": {
            "ml_very_high": 50,
            "ml_high": 40,
            "ml_moderate": 30,
            "extreme_deviation": 40,
            "high_deviation": 22
        }
    }


def analyze_transaction(tx_hash: str, transaction: Dict, model_timestamp: str = None) -> Dict:
    """
    Detailed analysis function for debugging
    Returns breakdown of all agent scores
    *** MODIFIED: Accepts and passes model_timestamp ***
    """
    # *** MODIFICATION: Pass timestamp to pipeline ***
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


# Test function
if __name__ == "__main__":
    print("Testing Agent Pipeline...")
    print("=" * 60)

    test_tx = {
        "tx_hash": "0xtest123",
        "from_address": "0xnorm000000000000000000000000000000000000",
        "to_address": "0xnorm000000000000000000000000000000000001",
        "value_eth": 2.5,
        "gas_price": 50.0,
        "timestamp": "2025-10-19T12:00:00Z",
        "is_fraud": 0
    }

    # Test with default model
    result = process_transaction_pipeline(test_tx)

    print(f"Final Status (Default Model): {result['final_status']}")
    print(f"Final Score: {result['final_score']}")
    print(f"\nAgent Scores:")
    print(f"  Agent 1: {result['agent_1_score']}")
    print(f"  Agent 2: {result['agent_2_score']}")
    print(f"  Agent 3: {result['agent_3_score']}")
    print(f"\nReasons: {result['reasons']}")
    print("=" * 60)
