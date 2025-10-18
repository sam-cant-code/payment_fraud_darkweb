"""
Agent Logic Module for Blockchain Fraud Detection MVP
Contains specialized agents and the processing pipeline
(Version with enhanced Agent 1 rules, adjusted thresholds, and is_fraud preservation)
"""

import pickle
import sqlite3
from typing import Dict, List
import os
from datetime import datetime, timedelta, UTC # Added UTC

# Import the neutralized network agent
try:
    from agent_3_network import agent_3_network
except ImportError:
    print("Warning: agent_3_network.py not found or contains errors. Agent 3 will be skipped.")
    def agent_3_network(transaction: Dict) -> Dict:
        transaction['agent_3_score'] = 0
        transaction['agent_3_reasons'] = ["Agent 3 skipped (module not found)"]
        return transaction

# --- Global Cache for Threat List (Load Once) ---
# Simple in-memory cache to avoid reading the file for every transaction
THREAT_LIST_CACHE = None
THREAT_LIST_LAST_LOADED = None
THREAT_LIST_FILEPATH = "data/dark_web_wallets.txt"

# --- Hypothetical known mixer/service addresses for Agent 1 ---
# These should match addresses added during setup if you want them flagged
KNOWN_MIXER_ADDRESSES = {
    f"0xmix{i:038x}" for i in range(2) # Match the generation in setup
}
# --- Track recent small transfers (Simple in-memory example) ---
# In production, use Redis or a proper cache
RECENT_SMALL_TRANSFERS = {} # Key: from_address, Value: [timestamp1, timestamp2, ...]
TRANSFER_WINDOW = timedelta(minutes=10) # Time window to check for frequency
TRANSFER_THRESHOLD = 5 # Number of small transfers in window to flag
SMALL_TRANSFER_VALUE = 0.01 # ETH value below which a transfer is considered "small" for frequency check

def load_dark_web_wallets(force_reload: bool = False) -> set:
    """Load the list of known malicious wallet addresses with caching"""
    global THREAT_LIST_CACHE, THREAT_LIST_LAST_LOADED
    now = datetime.now()
    # Reload if cache is empty, forced, or older than 5 minutes
    if THREAT_LIST_CACHE is None or force_reload or \
       (THREAT_LIST_LAST_LOADED and (now - THREAT_LIST_LAST_LOADED > timedelta(minutes=5))):

        if not os.path.exists(THREAT_LIST_FILEPATH):
             print(f"Warning: {THREAT_LIST_FILEPATH} not found. Threat list is empty. Run setup.")
             THREAT_LIST_CACHE = set()
        else:
            try:
                with open(THREAT_LIST_FILEPATH, 'r') as f:
                    wallets = set(line.strip().lower() for line in f if line.strip())
                THREAT_LIST_CACHE = wallets
                print(f"Loaded/Refreshed threat list: {len(THREAT_LIST_CACHE)} addresses.")
            except Exception as e:
                print(f"Error loading {THREAT_LIST_FILEPATH}: {e}. Using empty set.")
                THREAT_LIST_CACHE = set() # Ensure cache is a set even on error
        THREAT_LIST_LAST_LOADED = now

    # Ensure cache is initialized if it's still None (edge case)
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
    dark_web_wallets = load_dark_web_wallets() # Includes mixers now from setup script
    from_addr = transaction.get('from_address', '').lower()
    to_addr = transaction.get('to_address', '').lower()
    value_eth = transaction.get('value_eth', 0.0)
    timestamp_str = transaction.get('timestamp') # Get timestamp string

    # 1. Known Threat List Check
    if from_addr in dark_web_wallets:
        score += 50
        reasons.append(f"From-Wallet ({from_addr[:10]}...) on Threat List")
    if to_addr in dark_web_wallets:
        # Check if it's a known mixer specifically
        if to_addr in KNOWN_MIXER_ADDRESSES:
             score += 60 # Higher score for sending to mixer
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

    # 3. Zero Value Check (Often used for token approvals, but can be spam)
    if value_eth == 0.0 and transaction.get('gas_price', 0) > 0: # Ensure it's not a failed tx
         # Low score, might just be noise, but worth noting
         score += 2
         reasons.append("Zero ETH value transaction (potential token interaction or spam)")

    # 4. Small Repetitive Transfers (Dusting or Spam)
    if value_eth > 0 and value_eth < SMALL_TRANSFER_VALUE and from_addr:
        now = datetime.now(UTC) # Use timezone-aware now
        # Clean up old timestamps for this sender
        if from_addr in RECENT_SMALL_TRANSFERS:
            RECENT_SMALL_TRANSFERS[from_addr] = [
                ts for ts in RECENT_SMALL_TRANSFERS[from_addr] if now - ts <= TRANSFER_WINDOW
            ]
        else:
            RECENT_SMALL_TRANSFERS[from_addr] = []

        # Add current timestamp if available
        current_ts = None
        if timestamp_str:
            try:
                # Attempt to parse the timestamp string from transaction
                current_ts = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=UTC)
            except (ValueError, TypeError):
                 # Fallback to current time if parsing fails
                 current_ts = now

        if current_ts:
             RECENT_SMALL_TRANSFERS[from_addr].append(current_ts)

        # Check if threshold is met
        count = len(RECENT_SMALL_TRANSFERS[from_addr])
        if count >= TRANSFER_THRESHOLD:
            score += 15 # Add moderate score for high frequency small sends
            reasons.append(f"High frequency of small transfers ({count} in {TRANSFER_WINDOW}) from {from_addr[:10]}...")
            # Optional: Reset count for sender after flagging to avoid repeated flags immediately
            # RECENT_SMALL_TRANSFERS[from_addr] = []


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
    model_path = 'models/behavior_model.pkl'
    db_path = 'data/wallet_profiles.db'

    # --- ML Model Anomaly Detection ---
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            # Ensure features are in the same format/order as training
            features = [[
                transaction.get('value_eth', 0.0),
                transaction.get('gas_price', 0.0)
            ]]
            # score_samples returns the anomaly score (lower is more anomalous)
            anomaly_score_raw = model.score_samples(features)[0]
            # Convert anomaly score to a 0-50 risk contribution
            # Adjust scaling factor as needed. More negative scores are more anomalous.
            # Example scaling: score increases as anomaly_score_raw gets *more negative*
            anomaly_contribution = max(0, min(50, int((-anomaly_score_raw - 0.1) * 50))) # Heuristic scaling

            if anomaly_contribution > 0: # Only add score if considered anomalous
                score += anomaly_contribution
                reasons.append(f"ML Model: Behavioral anomaly detected (raw score: {anomaly_score_raw:.3f}, risk add: {anomaly_contribution})")
        except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, Exception) as e:
             print(f"Warning: Error loading/using ML model '{model_path}': {e}")
    else:
        print(f"Warning: Model file '{model_path}' not found. Skipping ML analysis. Run setup.")

    # --- Historical Profile Check ---
    if os.path.exists(db_path):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            from_addr = transaction.get('from_address', '')
            if from_addr: # Only check if from_address is valid
                cursor.execute("SELECT avg_transaction_value FROM wallet_profiles WHERE wallet_address = ?", (from_addr.lower(),)) # Ensure lowercase match
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

        except sqlite3.Error as e: print(f"Warning: DB error accessing wallet profiles '{db_path}': {e}")
        except Exception as e: print(f"Warning: Unexpected error accessing wallet profiles: {e}")
        finally:
            if conn: conn.close()
    else:
        print(f"Warning: Wallet profiles DB '{db_path}' not found. Skipping historical checks. Run setup.")

    transaction['agent_2_score'] = score
    transaction['agent_2_reasons'] = reasons
    return transaction


def agent_4_decision(transaction: Dict) -> Dict:
    """
    Agent 4: Decision Aggregator (Adjusted Thresholds)
    - Combines scores from agents 1 and 2. Agent 3 is neutralized.
    - Applies LOWERED thresholds for increased sensitivity.
    - Makes final decision on transaction status.
    """
    agent_1_score = transaction.get('agent_1_score', 0)
    agent_2_score = transaction.get('agent_2_score', 0)
    agent_3_score = transaction.get('agent_3_score', 0) # Ignored in calculation

    # Final score based only on Agents 1 & 2
    final_score = (agent_1_score * 1.5) + (agent_2_score * 1.0) # Agent 3 weight removed

    all_reasons = []
    all_reasons.extend(transaction.get('agent_1_reasons', []))
    all_reasons.extend(transaction.get('agent_2_reasons', []))
    all_reasons.extend(transaction.get('agent_3_reasons', [])) # Keep neutral reason for clarity

    # Adjusted score cap (max possible from A1*1.5 + A2*1 is roughly 125, let's use 130)
    final_score = round(min(final_score, 130.0), 1) # Round final score

    # --- LOWERED THRESHOLDS ---
    DENY_THRESHOLD = 50 # Lowered from 80
    REVIEW_THRESHOLD = 25 # Lowered from 40
    # --- END THRESHOLD ADJUSTMENT ---

    if final_score >= DENY_THRESHOLD:
        status = "DENY"
    elif final_score >= REVIEW_THRESHOLD:
        status = "FLAG_FOR_REVIEW"
    else:
        status = "APPROVE"

    transaction['final_score'] = final_score
    transaction['final_status'] = status
    # Provide a default reason if approved and no specific reasons were generated
    if not all_reasons and status == "APPROVE":
         transaction['reasons'] = "No specific flags detected by agents."
    elif not all_reasons: # Should not happen if score > 0, but safety check
         transaction['reasons'] = "No specific reasons listed."
    else:
         transaction['reasons'] = " | ".join(all_reasons)

    return transaction


# --- PIPELINE FUNCTION ---

def process_transaction_pipeline(transaction: Dict) -> Dict:
    """
    Pass transaction through the complete agent pipeline:
    Agent 1 (Monitor Enhanced) -> Agent 2 (Behavioral) -> Agent 3 (Network - Neutralized) -> Agent 4 (Decision Adjusted)
    """
    # Preserve the original 'is_fraud' label if it exists
    is_fraud_label = transaction.get('is_fraud') # Get label before processing

    # Agent 1: Threat Intelligence (Enhanced)
    processed_tx = agent_1_monitor(transaction.copy()) # Pass a copy

    # Agent 2: Behavioral Analysis
    processed_tx = agent_2_behavioral(processed_tx)

    # Agent 3: Network Analysis (Neutralized)
    processed_tx = agent_3_network(processed_tx)

    # Agent 4: Final Decision (Adjusted Thresholds)
    processed_tx = agent_4_decision(processed_tx)

    # --- ADD is_fraud back to the final dictionary ---
    if is_fraud_label is not None:
        processed_tx['is_fraud'] = is_fraud_label
    # --- END CHANGE ---

    return processed_tx