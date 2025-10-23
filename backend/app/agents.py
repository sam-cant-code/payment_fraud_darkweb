import os
import sqlite3
from pathlib import Path
import requests # <-- ADDED
import time      # <-- ADDED
from dotenv import load_dotenv # <-- ADDED

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / 'data' / 'wallet_profiles.db' # Still needed for Agent 2 (local cache)
THREAT_LIST_PATH = PROJECT_ROOT / 'data' / 'dark_web_wallets.txt'

# --- Load Environment Variables ---
load_dotenv(PROJECT_ROOT / 'backend' / '.env') # <-- ADDED: Load .env from backend folder
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY") # <-- ADDED
ETHERSCAN_API_URL = "https://api.etherscan.io/api" # <-- ADDED

if not ETHERSCAN_API_KEY:
    print("="*50)
    print("WARNING: ETHERSCAN_API_KEY not found in .env file.")
    print("Agent 2 (Wallet Monitor) will not work with live data.")
    print("="*50)


# --- Agent 1: Transaction Monitor ---
class Agent_1_Transaction_Monitor:
    """Analyzes intrinsic transaction properties like value and gas."""
    
    def analyze(self, transaction_data):
        score = 10  # Start with a low base score
        reasons = []
        
        tx_value = float(transaction_data.get('value_eth', 0))
        gas_price = float(transaction_data.get('gas_price', 0)) # Assuming gas_price is in Gwei

        # 1. Value Check
        if tx_value > 10:
            score = 80 # High value is a strong indicator
            reasons.append(f"High value: {tx_value:.2f} ETH")
        else:
            # GOOD REASON: Value is normal
            reasons.append(f"Normal value: {tx_value:.2f} ETH")

        # 2. Gas Price Check
        if gas_price > 100: # e.g., > 100 Gwei
            score = max(score, 60) # High gas is a medium indicator
            reasons.append(f"High gas: {gas_price:.0f} Gwei")
        else:
            # GOOD REASON: Gas price is normal
            reasons.append(f"Normal gas: {gas_price:.0f} Gwei")
            
        return {"score": score, "reasons": reasons}

# --- Agent 2: Wallet Monitor (REWRITTEN) ---
class Agent_2_Wallet_Monitor:
    """Analyzes wallet profiles using LIVE Etherscan data."""
    
    def __init__(self):
        # We no longer use the DB, but a cache is smart to avoid
        # hitting the API for the same address repeatedly.
        self.profile_cache = {}
        print("Agent 2: Initialized (Live Etherscan Mode)")

    def get_wallet_profile_from_etherscan(self, wallet_address):
        """
        Fetches live wallet data from Etherscan API.
        Returns a dict: {'first_tx_timestamp': int, 'tx_count': int} or None
        """
        # Check cache first
        if wallet_address in self.profile_cache:
            return self.profile_cache[wallet_address]
            
        if not ETHERSCAN_API_KEY:
            print("Agent 2: No Etherscan API key. Skipping profile check.")
            return None

        profile = {}
        
        try:
            # 1. Get Transaction Count (Nonce)
            params_tx_count = {
                'module': 'proxy',
                'action': 'eth_getTransactionCount',
                'address': wallet_address,
                'tag': 'latest',
                'apikey': ETHERSCAN_API_KEY
            }
            resp_tx_count = requests.get(ETHERSCAN_API_URL, params=params_tx_count, timeout=5)
            resp_tx_count.raise_for_status() # Raise error on bad status
            
            data_tx_count = resp_tx_count.json()
            if data_tx_count.get('result'):
                # Result is in Hex (e.g., "0x1a"), convert to int
                profile['tx_count'] = int(data_tx_count['result'], 16)
            else:
                profile['tx_count'] = 0

            # 2. Get First Transaction (for Age)
            params_first_tx = {
                'module': 'account',
                'action': 'txlist',
                'address': wallet_address,
                'startblock': 0,
                'endblock': 99999999,
                'page': 1,
                'offset': 1,
                'sort': 'asc',
                'apikey': ETHERSCAN_API_KEY
            }
            resp_first_tx = requests.get(ETHERSCAN_API_URL, params=params_first_tx, timeout=5)
            resp_first_tx.raise_for_status()

            data_first_tx = resp_first_tx.json()
            # 'result' is a list of transactions
            if data_first_tx.get('status') == '1' and data_first_tx.get('result'):
                first_tx = data_first_tx['result'][0]
                profile['first_tx_timestamp'] = int(first_tx['timeStamp'])
            else:
                # No transactions found, wallet is new. Set timestamp to now.
                profile['first_tx_timestamp'] = int(time.time())

            # Save to cache and return
            self.profile_cache[wallet_address] = profile
            return profile

        except requests.exceptions.RequestException as e:
            print(f"Etherscan API Error for {wallet_address}: {e}")
            # If API fails, return None so we can handle it gracefully
            return None

    def analyze(self, transaction_data):
        score = 10 # Start low
        reasons = []
        
        from_address = transaction_data.get('from_address')
        to_address = transaction_data.get('to_address')

        from_profile = self.get_wallet_profile_from_etherscan(from_address)
        to_profile = self.get_wallet_profile_from_etherscan(to_address)

        current_time_seconds = int(time.time())

        # 1. Analyze Sender Profile
        if from_profile:
            age_seconds = current_time_seconds - from_profile['first_tx_timestamp']
            age_days = age_seconds / 86400 # 60*60*24
            tx_count = from_profile['tx_count']

            # Check age
            if age_days < 1:
                score = 90
                reasons.append(f"Sender wallet new ({age_days:.1f} days)")
            else:
                reasons.append(f"Sender wallet age OK ({age_days:.1f} days)")
            
            # Check transaction count (nonce)
            if tx_count <= 1: # A count of 0 or 1 is very suspicious for a sender
                score = max(score, 85)
                reasons.append(f"Sender has low tx count ({tx_count})")
            else:
                reasons.append(f"Sender tx count OK ({tx_count})")
        else:
            score = max(score, 30) # API error or invalid address is a slight risk
            reasons.append("Sender profile unknown (API error or new wallet)")

        # 2. Analyze Receiver Profile (less strict)
        if to_profile:
            age_seconds = current_time_seconds - to_profile['first_tx_timestamp']
            age_days = age_seconds / 86400
            
            if age_days < 1:
                score = max(score, 70) # Receiver being new is also risky
                reasons.append(f"Receiver wallet new ({age_days:.1f} days)")
            else:
                reasons.append(f"Receiver wallet age OK ({age_days:.1f} days)")
        else:
            score = max(score, 30) # Unknown receiver
            reasons.append("Receiver profile unknown (API error or new wallet)")

        return {"score": score, "reasons": reasons}


# --- Agent 3: Threat Intel ---
class Agent_3_Threat_Intel:
    """Checks addresses against a known threat list."""
    
    def __init__(self, threat_list_path=THREAT_LIST_PATH):
        self.threat_list_path = str(threat_list_path)
        self.threat_list = self.load_threat_list()

    def load_threat_list(self):
        """Loads the threat list from a file into a set for fast lookup."""
        threat_set = set()
        if not os.path.exists(self.threat_list_path):
            print(f"Warning: Threat list not found at {self.threat_list_path}. Agent 3 will be ineffective.")
            return threat_set
            
        try:
            with open(self.threat_list_path, 'r') as f:
                for line in f:
                    threat_set.add(line.strip().lower())
            print(f"Loaded {len(threat_set)} addresses into threat intel list.")
        except Exception as e:
            print(f"Error loading threat list: {e}")
        return threat_set

    def analyze(self, transaction_data):
        score = 10
        reasons = []
        
        from_address = transaction_data.get('from_address', '').lower()
        to_address = transaction_data.get('to_address', '').lower()

        # 1. Check Sender
        if from_address in self.threat_list:
            score = 100
            reasons.append("SENDER on threat list")
        else:
            reasons.append("Sender not on threat list")

        # 2. Check Receiver
        if to_address in self.threat_list:
            score = 100
            reasons.append("RECEIVER on threat list")
        else:
            reasons.append("Receiver not on threat list")

        return {"score": score, "reasons": reasons}

# --- Main Pipeline ---

# Initialize agents (global instances for efficiency)
agent_1 = Agent_1_Transaction_Monitor()
agent_2 = Agent_2_Wallet_Monitor() # <-- MODIFIED: No longer needs DB_PATH
agent_3 = Agent_3_Threat_Intel(THREAT_LIST_PATH)

def process_transaction_pipeline(transaction_data):
    """
    Runs a transaction through the multi-agent analysis pipeline.
    """
    
    if not isinstance(transaction_data, dict):
        print(f"Error: Invalid input to pipeline. Expected dict, got {type(transaction_data)}")
        return None

    processed_tx = transaction_data.copy()
    reasons = []
    
    # --- Run Agents ---
    try:
        agent_1_results = agent_1.analyze(processed_tx)
        processed_tx['agent_1_score'] = agent_1_results['score']
        reasons.extend(agent_1_results['reasons'])
    except Exception as e:
        print(f"Error in Agent 1: {e}")
        processed_tx['agent_1_score'] = 0
        reasons.append("Agent 1 Error")

    try:
        agent_2_results = agent_2.analyze(processed_tx) # <-- This now calls the new live agent
        processed_tx['agent_2_score'] = agent_2_results['score']
        reasons.extend(agent_2_results['reasons'])
    except Exception as e:
        print(f"Error in Agent 2: {e}")
        processed_tx['agent_2_score'] = 0
        reasons.append("Agent 2 Error")

    try:
        agent_3_results = agent_3.analyze(processed_tx)
        processed_tx['agent_3_score'] = agent_3_results['score']
        reasons.extend(agent_3_results['reasons'])
    except Exception as e:
        print(f"Error in Agent 3: {e}")
        processed_tx['agent_3_score'] = 0
        reasons.append("Agent 3 Error")

    # --- Final Aggregation ---
    scores = [
        processed_tx.get('agent_1_score', 0),
        processed_tx.get('agent_2_score', 0),
        processed_tx.get('agent_3_score', 0)
    ]
    
    if 100 in scores:
        final_score = 100
    else:
        final_score = sum(scores) / len(scores) if scores else 0

    # Determine final status
    if final_score >= 90:
        final_status = "DENY"
    elif final_score >= 60:
        final_status = "FLAG_FOR_REVIEW"
    else:
        final_status = "APPROVE"

    processed_tx['reasons'] = "; ".join(reasons)
    processed_tx['final_score'] = final_score
    processed_tx['final_status'] = final_status

    if 'ml_fraud_probability' not in processed_tx:
         processed_tx['ml_fraud_probability'] = None
         
    if 'is_fraud' not in processed_tx:
        processed_tx['is_fraud'] = None

    return processed_tx