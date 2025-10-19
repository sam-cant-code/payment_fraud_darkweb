"""
Agent Logic Module for Blockchain Fraud Detection MVP
Contains specialized agents and the processing pipeline
(Updated to use supervised learning model with proper feature engineering)
"""

import pickle
import sqlite3
from typing import Dict, List
import os
from datetime import datetime, timedelta, UTC
import numpy as np

# Import the neutralized network agent
try:
    from agent_3_network import agent_3_network
except ImportError:
    print("Warning: agent_3_network.py not found or contains errors. Agent 3 will be skipped.")
    def agent_3_network(transaction: Dict) -> Dict:
        transaction['agent_3_score'] = 0
        transaction['agent_3_reasons'] = ["Agent 3 skipped (module not found)"]
        return transaction

# --- Global Cache for Threat List ---
THREAT_LIST_CACHE = None
THREAT_LIST_LAST_LOADED = None
# Note: THREAT_LIST_FILEPATH is now calculated dynamically in load_dark_web_wallets()

# --- Known mixer/service addresses ---
KNOWN_MIXER_ADDRESSES = {
    f"0xmix{i:038x}" for i in range(2)
}

# --- Track recent small transfers ---
RECENT_SMALL_TRANSFERS = {}
TRANSFER_WINDOW = timedelta(minutes=10)
TRANSFER_THRESHOLD = 5
SMALL_TRANSFER_VALUE = 0.01

def load_dark_web_wallets(force_reload: bool = False) -> set:
    """Load the list of known malicious wallet addresses with caching"""
    global THREAT_LIST_CACHE, THREAT_LIST_LAST_LOADED
    now = datetime.now()
    
    # Get absolute path to threat list
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    threat_filepath = os.path.join(backend_dir, 'data', 'dark_web_wallets.txt')
    
    if THREAT_LIST_CACHE is None or force_reload or \
       (THREAT_LIST_LAST_LOADED and (now - THREAT_LIST_LAST_LOADED > timedelta(minutes=5))):
        
        if not os.path.exists(threat_filepath):
            print(f"Warning: {threat_filepath} not found. Threat list is empty. Run setup.")
            THREAT_LIST_CACHE = set()
        else:
            try:
                with open(threat_filepath, 'r') as f:
                    wallets = set(line.strip().lower() for line in f if line.strip())
                THREAT_LIST_CACHE = wallets
                print(f"Loaded/Refreshed threat list: {len(THREAT_LIST_CACHE)} addresses.")
            except Exception as e:
                print(f"Error loading {threat_filepath}: {e}. Using empty set.")
                THREAT_LIST_CACHE = set()
        THREAT_LIST_LAST_LOADED = now
    
    if THREAT_LIST_CACHE is None:
        THREAT_LIST_CACHE = set()
    
    return THREAT_LIST_CACHE


def agent_1_monitor(transaction: Dict) -> Dict:
    """
    Agent 1: Threat Intelligence Engine (Enhanced Rules)
    - Checks against dark web/mixer lists
    - Applies stateless rules (high value, zero value, small repetitive)
    """
    score = 0
    reasons = []
    dark_web_wallets = load_dark_web_wallets()
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value_eth = transaction.get('value_eth', 0.0)
    timestamp_str = transaction.get('timestamp')

    # 1. Known Threat List Check
    if from_addr in dark_web_wallets:
        score += 50
        reasons.append(f"From-Wallet ({from_addr[:10]}...) on Threat List")
    if to_addr in dark_web_wallets:
        if to_addr in KNOWN_MIXER_ADDRESSES:
            score += 60
            reasons.append(f"To-Wallet ({to_addr[:10]}...) is a known Mixer")
        else:
            score += 50
            reasons.append(f"To-Wallet ({to_addr[:10]}...) on Threat List")

    # 2. High Value Check
    if value_eth > 100:
        score += 30
        reasons.append(f"Very high-value transaction: {value_eth:.4f} ETH")
    elif value_eth > 50:
        score += 20
        reasons.append(f"High-value transaction: {value_eth:.4f} ETH")

    # 3. Zero Value Check
    if value_eth == 0.0 and transaction.get('gas_price', 0) > 0:
        score += 2
        reasons.append("Zero ETH value transaction (potential token interaction or spam)")

    # 4. Small Repetitive Transfers
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
            score += 15
            reasons.append(f"High frequency of small transfers ({count} in {TRANSFER_WINDOW}) from {from_addr[:10]}...")

    transaction['agent_1_score'] = score
    transaction['agent_1_reasons'] = reasons
    return transaction


def engineer_features_for_prediction(transaction: Dict) -> np.ndarray:
    """
    Engineer features for a single transaction to match training format
    Returns feature array ready for model prediction
    """
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value = transaction.get('value_eth', 0.0)
    gas = transaction.get('gas_price', 0.0)
    
    # Load threat list
    threat_set = set(w.lower() for w in load_dark_web_wallets())
    
    # Basic features
    features = [
        value,  # Transaction value
        gas,    # Gas price
        value * gas,  # Total cost
    ]
    
    # Threat intelligence features
    features.append(1 if from_addr in threat_set else 0)
    features.append(1 if to_addr in threat_set else 0)
    
    # Wallet behavior features (simplified for single transaction)
    # Get absolute path to database
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    db_path = os.path.join(backend_dir, 'data', 'wallet_profiles.db')
    
    sender_tx_count = 0
    sender_avg_value = 0.1  # Default
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
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
        except:
            pass
    
    features.append(sender_tx_count)
    features.append(sender_avg_value)
    features.append(value / max(sender_avg_value, 0.001))  # Deviation ratio
    
    # Time-based features
    try:
        ts = datetime.fromisoformat(transaction.get('timestamp', '').replace('Z', ''))
        features.append(ts.hour)
        features.append(ts.weekday())
    except:
        features.append(12)  # Default hour
        features.append(3)   # Default day
    
    features.append(86400)  # Default velocity (1 day in seconds)
    
    # Value-based risk indicators
    features.append(1 if value > 100 else 0)
    features.append(1 if value > 50 else 0)
    features.append(1 if gas > 150 else 0)
    
    # Ratio features
    features.append(gas / max(value, 0.001))
    
    return np.array(features).reshape(1, -1)


def agent_2_behavioral(transaction: Dict) -> Dict:
    """
    Agent 2: Behavioral Profiler (UPDATED TO USE SUPERVISED MODEL)
    - Uses ML model for fraud prediction
    - Checks transaction against wallet's historical behavior
    """
    score = 0
    reasons = []
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)  # Go up to backend/
    
    model_path = os.path.join(backend_dir, 'models', 'fraud_model.pkl')
    scaler_path = os.path.join(backend_dir, 'models', 'scaler.pkl')
    db_path = os.path.join(backend_dir, 'data', 'wallet_profiles.db')

    # --- ML Model Prediction (SUPERVISED LEARNING) ---
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # Load model and scaler
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Engineer features
            features = engineer_features_for_prediction(transaction)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Get prediction and probability
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            fraud_probability = probability[1] if len(probability) > 1 else 0.0
            
            # Convert probability to risk score (0-50 scale)
            ml_risk_score = int(fraud_probability * 50)
            
            if prediction == 1 or fraud_probability > 0.5:
                score += ml_risk_score
                reasons.append(f"ML Model: High fraud probability ({fraud_probability:.2%}, risk: {ml_risk_score})")
            elif fraud_probability > 0.3:
                score += int(ml_risk_score * 0.7)
                reasons.append(f"ML Model: Moderate fraud probability ({fraud_probability:.2%})")
                
        except Exception as e:
            print(f"Warning: Error using ML model: {e}")
    else:
        print(f"Warning: Model files not found. Skipping ML analysis. Run setup.")

    # --- Historical Profile Check ---
    if os.path.exists(db_path):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
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
                            score += 30
                            reasons.append(f"History: Value {deviation_ratio:.1f}x higher than wallet avg ({avg_value:.4f} ETH)")
                        elif deviation_ratio > 5:
                            score += 15
                            reasons.append(f"History: Value {deviation_ratio:.1f}x higher than wallet avg ({avg_value:.4f} ETH)")
                    elif current_value > 1:
                        score += 10
                        reasons.append(f"History: Significant value ({current_value:.4f} ETH) from wallet with near-zero avg history")
        except Exception as e:
            print(f"Warning: Error accessing wallet profiles: {e}")
        finally:
            if conn:
                conn.close()

    transaction['agent_2_score'] = score
    transaction['agent_2_reasons'] = reasons
    return transaction


def agent_4_decision(transaction: Dict) -> Dict:
    """
    Agent 4: Decision Aggregator
    - Combines scores from agents 1 and 2
    - Makes final decision on transaction status
    """
    agent_1_score = transaction.get('agent_1_score', 0)
    agent_2_score = transaction.get('agent_2_score', 0)
    agent_3_score = transaction.get('agent_3_score', 0)

    # Final score based on Agents 1 & 2
    final_score = (agent_1_score * 1.5) + (agent_2_score * 1.0)

    all_reasons = []
    all_reasons.extend(transaction.get('agent_1_reasons', []))
    all_reasons.extend(transaction.get('agent_2_reasons', []))
    all_reasons.extend(transaction.get('agent_3_reasons', []))

    final_score = round(min(final_score, 150.0), 1)

    # Thresholds
    DENY_THRESHOLD = 50
    REVIEW_THRESHOLD = 25

    if final_score >= DENY_THRESHOLD:
        status = "DENY"
    elif final_score >= REVIEW_THRESHOLD:
        status = "FLAG_FOR_REVIEW"
    else:
        status = "APPROVE"

    transaction['final_score'] = final_score
    transaction['final_status'] = status
    
    if not all_reasons and status == "APPROVE":
        transaction['reasons'] = "No specific flags detected by agents."
    elif not all_reasons:
        transaction['reasons'] = "No specific reasons listed."
    else:
        transaction['reasons'] = " | ".join(all_reasons)

    return transaction


def process_transaction_pipeline(transaction: Dict) -> Dict:
    """
    Pass transaction through the complete agent pipeline
    """
    is_fraud_label = transaction.get('is_fraud')

    processed_tx = agent_1_monitor(transaction.copy())
    processed_tx = agent_2_behavioral(processed_tx)
    processed_tx = agent_3_network(processed_tx)
    processed_tx = agent_4_decision(processed_tx)

    if is_fraud_label is not None:
        processed_tx['is_fraud'] = is_fraud_label

    return processed_tx