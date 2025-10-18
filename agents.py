"""
Agent Logic Module for Blockchain Fraud Detection MVP
Contains three specialized agents for transaction analysis
"""

import pickle
import sqlite3
from typing import Dict, List


def load_dark_web_wallets(filepath: str = "data/dark_web_wallets.txt") -> set:
    """Load the list of known malicious wallet addresses"""
    try:
        with open(filepath, 'r') as f:
            wallets = set(line.strip().lower() for line in f if line.strip())
        return wallets
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Returning empty set.")
        return set()


def agent_1_monitor(transaction: Dict) -> Dict:
    """
    Agent 1: Threat Intelligence Engine
    - Checks against dark web wallet list
    - Applies stateless rules for high-value transactions
    """
    score = 0
    reasons = []
    
    # Load threat intelligence
    dark_web_wallets = load_dark_web_wallets()
    
    # Check if sender or receiver is on dark web list
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    
    if from_addr in dark_web_wallets:
        score += 50
        reasons.append(f"From-Wallet ({from_addr[:10]}...) on Dark Web Threat List")
    
    if to_addr in dark_web_wallets:
        score += 50
        reasons.append(f"To-Wallet ({to_addr[:10]}...) on Dark Web Threat List")
    
    # High-value transaction rule
    value_eth = transaction.get('value_eth', 0)
    if value_eth > 50:
        score += 20
        reasons.append(f"High-value transaction: {value_eth} ETH")
    
    # Very high-value transaction
    if value_eth > 100:
        score += 30
        reasons.append(f"Very high-value transaction: {value_eth} ETH")
    
    # Add scores and reasons to transaction
    transaction['agent_1_score'] = score
    transaction['agent_1_reasons'] = reasons
    
    return transaction


def agent_2_behavioral(transaction: Dict) -> Dict:
    """
    Agent 2: Behavioral Profiler
    - Uses ML model to detect anomalies
    - Checks transaction against wallet's historical behavior
    """
    score = 0
    reasons = []
    
    # Load the trained behavior model
    try:
        with open('models/behavior_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Prepare features for the model
        features = [[
            transaction.get('value_eth', 0),
            transaction.get('gas_price', 0)
        ]]
        
        # Get anomaly prediction (-1 = anomaly, 1 = normal)
        prediction = model.predict(features)[0]
        anomaly_score = model.score_samples(features)[0]
        
        if prediction == -1:
            score += 25
            reasons.append(f"ML Model detected anomaly (score: {anomaly_score:.3f})")
    
    except FileNotFoundError:
        print("Warning: behavior_model.pkl not found. Skipping ML analysis.")
    except Exception as e:
        print(f"Warning: Error loading model: {e}")
    
    # Check against wallet historical behavior
    try:
        conn = sqlite3.connect('data/wallet_profiles.db')
        cursor = conn.cursor()
        
        from_addr = transaction.get('from_address', '')
        cursor.execute("""
            SELECT avg_transaction_value, transaction_count 
            FROM wallet_profiles 
            WHERE wallet_address = ?
        """, (from_addr,))
        
        result = cursor.fetchone()
        
        if result:
            avg_value, tx_count = result
            current_value = transaction.get('value_eth', 0)
            
            # Check if transaction is 10x higher than historical average
            if avg_value > 0 and current_value > (avg_value * 10):
                score += 30
                reasons.append(f"Transaction 10x higher than wallet average ({avg_value:.2f} ETH)")
            
            # Check if transaction is significantly higher (5x)
            elif avg_value > 0 and current_value > (avg_value * 5):
                score += 15
                reasons.append(f"Transaction 5x higher than wallet average ({avg_value:.2f} ETH)")
        
        conn.close()
    
    except sqlite3.Error as e:
        print(f"Warning: Database error: {e}")
    except FileNotFoundError:
        print("Warning: wallet_profiles.db not found.")
    
    # Add scores and reasons to transaction
    transaction['agent_2_score'] = score
    transaction['agent_2_reasons'] = reasons
    
    return transaction


def agent_4_decision(transaction: Dict) -> Dict:
    """
    Agent 4: Decision Aggregator
    - Combines scores from all agents
    - Makes final decision on transaction status
    """
    # Get scores from previous agents
    agent_1_score = transaction.get('agent_1_score', 0)
    agent_2_score = transaction.get('agent_2_score', 0)
    
    # Calculate final score with weighted importance
    # Agent 1 (threat intel) gets higher weight
    final_score = (agent_1_score * 1.5) + agent_2_score
    
    # Combine all reasons
    all_reasons = []
    all_reasons.extend(transaction.get('agent_1_reasons', []))
    all_reasons.extend(transaction.get('agent_2_reasons', []))
    
    # Make final decision based on score thresholds
    if final_score >= 70:
        status = "DENY"
    elif final_score >= 30:
        status = "FLAG_FOR_REVIEW"
    else:
        status = "APPROVE"
    
    # Add final decision to transaction
    transaction['final_score'] = final_score
    transaction['final_status'] = status
    transaction['reasons'] = " | ".join(all_reasons) if all_reasons else "No issues detected"
    
    return transaction