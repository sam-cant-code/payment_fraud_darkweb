"""
Agent Logic Module for Blockchain Fraud Detection MVP
Contains specialized agents and the processing pipeline
"""

import pickle
import sqlite3
from typing import Dict, List
import os # Added for path checking

# Import the network agent (assuming agent_3_network.py exists)
try:
    from agent_3_network import agent_3_network
except ImportError:
    print("Warning: agent_3_network.py not found or contains errors. Agent 3 will be skipped.")
    # Define a dummy function if agent 3 is missing
    def agent_3_network(transaction: Dict) -> Dict:
        transaction['agent_3_score'] = 0
        transaction['agent_3_reasons'] = ["Agent 3 skipped (module not found)"]
        return transaction

# --- Individual Agent Functions ---

def load_dark_web_wallets(filepath: str = "data/dark_web_wallets.txt") -> set:
    """Load the list of known malicious wallet addresses"""
    # ... (function remains the same as previous correct version) ...
    if not os.path.exists(filepath):
         print(f"Warning: {filepath} not found. Returning empty set. Run setup.")
         return set()
    try:
        with open(filepath, 'r') as f:
            wallets = set(line.strip().lower() for line in f if line.strip())
        return wallets
    except Exception as e:
        print(f"Error loading {filepath}: {e}. Returning empty set.")
        return set()

def agent_1_monitor(transaction: Dict) -> Dict:
    """
    Agent 1: Threat Intelligence Engine
    - Checks against dark web wallet list
    - Applies stateless rules for high-value transactions
    """
    # ... (function remains the same as previous correct version) ...
    score = 0
    reasons = []
    dark_web_wallets = load_dark_web_wallets()
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    if from_addr in dark_web_wallets:
        score += 50
        reasons.append(f"From-Wallet ({from_addr[:10]}...) on Dark Web Threat List")
    if to_addr in dark_web_wallets:
        score += 50
        reasons.append(f"To-Wallet ({to_addr[:10]}...) on Dark Web Threat List")
    value_eth = transaction.get('value_eth', 0)
    if value_eth > 100:
        score += 30
        reasons.append(f"Very high-value transaction: {value_eth:.4f} ETH")
    elif value_eth > 50: # Check this *after* very high value
        score += 20
        reasons.append(f"High-value transaction: {value_eth:.4f} ETH")
    transaction['agent_1_score'] = score
    transaction['agent_1_reasons'] = reasons
    return transaction

def agent_2_behavioral(transaction: Dict) -> Dict:
    """
    Agent 2: Behavioral Profiler
    - Uses ML model to detect anomalies
    - Checks transaction against wallet's historical behavior
    """
    # ... (function remains the same as previous correct version) ...
    score = 0
    reasons = []
    model_path = 'models/behavior_model.pkl'
    db_path = 'data/wallet_profiles.db'

    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            features = [[ transaction.get('value_eth', 0), transaction.get('gas_price', 0) ]]
            prediction = model.predict(features)[0]
            anomaly_score = model.score_samples(features)[0]
            if prediction == -1:
                anomaly_intensity_score = max(0, min(50, int(abs(anomaly_score) * 10)))
                score += anomaly_intensity_score
                reasons.append(f"ML Model detected behavioral anomaly (score: {anomaly_score:.3f}, added: {anomaly_intensity_score})")
        except Exception as e: print(f"Warning: Error loading/using ML model: {e}")
    else: print(f"Warning: {model_path} not found. Skipping ML analysis. Run setup.")

    if os.path.exists(db_path):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            from_addr = transaction.get('from_address', '')
            cursor.execute("SELECT avg_transaction_value FROM wallet_profiles WHERE wallet_address = ?", (from_addr,))
            result = cursor.fetchone()
            if result and result[0] is not None:
                avg_value = result[0]
                current_value = transaction.get('value_eth', 0)
                if avg_value > 0:
                    deviation_ratio = current_value / avg_value
                    if deviation_ratio > 10:
                        score += 30
                        reasons.append(f"Value {deviation_ratio:.1f}x higher than wallet avg ({avg_value:.2f} ETH)")
                    elif deviation_ratio > 5:
                        score += 15
                        reasons.append(f"Value {deviation_ratio:.1f}x higher than wallet avg ({avg_value:.2f} ETH)")
        except sqlite3.Error as e: print(f"Warning: DB error accessing wallet profiles: {e}")
        except Exception as e: print(f"Warning: Unexpected error accessing wallet profiles: {e}")
        finally:
            if conn: conn.close()
    else: print(f"Warning: {db_path} not found. Skipping historical checks. Run setup.")

    transaction['agent_2_score'] = score
    transaction['agent_2_reasons'] = reasons
    return transaction

# Agent 3 is imported

def agent_4_decision(transaction: Dict) -> Dict:
    """
    Agent 4: Decision Aggregator
    - Combines scores from all agents (1, 2, and 3)
    - Makes final decision on transaction status
    """
    # ... (function remains the same as previous correct version) ...
    agent_1_score = transaction.get('agent_1_score', 0)
    agent_2_score = transaction.get('agent_2_score', 0)
    agent_3_score = transaction.get('agent_3_score', 0)
    final_score = (agent_1_score * 1.5) + (agent_2_score * 1.0) + (agent_3_score * 1.2)
    all_reasons = []
    all_reasons.extend(transaction.get('agent_1_reasons', []))
    all_reasons.extend(transaction.get('agent_2_reasons', []))
    all_reasons.extend(transaction.get('agent_3_reasons', []))
    final_score = min(final_score, 150.0) # Cap score
    if final_score >= 80: status = "DENY"
    elif final_score >= 40: status = "FLAG_FOR_REVIEW"
    else: status = "APPROVE"
    transaction['final_score'] = final_score
    transaction['final_status'] = status
    transaction['reasons'] = " | ".join(all_reasons) if all_reasons else "No specific issues detected"
    return transaction

# --- MOVED PIPELINE FUNCTION ---

def process_transaction_pipeline(transaction: Dict) -> Dict:
    """
    Pass transaction through the complete agent pipeline:
    Agent 1 (Monitor) -> Agent 2 (Behavioral) -> Agent 3 (Network) -> Agent 4 (Decision)
    """
    # Agent 1: Threat Intelligence
    transaction = agent_1_monitor(transaction)

    # Agent 2: Behavioral Analysis
    transaction = agent_2_behavioral(transaction)

    # Agent 3: Network Analysis
    transaction = agent_3_network(transaction) # Uses the imported or dummy function

    # Agent 4: Final Decision (Considers Agent 3)
    transaction = agent_4_decision(transaction)

    return transaction