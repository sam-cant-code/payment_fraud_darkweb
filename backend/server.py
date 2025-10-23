import csv
import json
import os
import subprocess
import sqlite3
import pandas as pd
import time # <-- Already present, needed for new API
import random
import threading
from datetime import datetime, UTC 
from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from web3 import Web3
from web3.providers import HTTPProvider
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
# Make sure your .env file is in the 'backend' folder
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Import agent logic
# --- MODIFIED IMPORT ---
from app.agents import process_transaction_pipeline, agent_2 # <-- Use the live agent_2

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/socket.io/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Path Definitions (Relative to this server.py file) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'output', 'all_processed_transactions.csv')
DB_FILE = os.path.join(PROJECT_ROOT, 'data', 'wallet_profiles.db') # Still used for /api/status check
THREAT_FILE = os.path.join(PROJECT_ROOT, 'data', 'dark_web_wallets.txt')
MODEL_FILE = os.path.join(PROJECT_ROOT, 'models', 'behavior_model.pkl')
PYTHON_EXE = 'python'

# --- Alchemy Configuration ---
ALCHEMY_HTTP_URL = os.environ.get("ALCHEMY_HTTP_URL", "YOUR_ALCHEMY_HTTP_URL")
if ALCHEMY_HTTP_URL == "YOUR_ALCHEMY_HTTP_URL":
    print("*" * 60)
    print("WARNING: ALCHEMY_HTTP_URL is not set in .env or environment.")
    print("Real-time feed will not work.")
    print("*" * 60)

# --- Global variables ---
alchemy_listener_running = False
alchemy_thread = None
w3 = None

# --- CSV Fieldnames (including is_fraud) ---
CSV_FIELDNAMES = [
    'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
    'timestamp', 'final_status', 'final_score', 'reasons',
    'agent_1_score', 'agent_2_score', 'agent_3_score',
    'ml_fraud_probability', # Added from simulation
    'is_fraud' # Added ground truth label column
]


# --- Helper Functions ---
def read_threat_list():
    if not os.path.exists(THREAT_FILE): return []
    try:
        with open(THREAT_FILE, 'r') as f: return [line.strip().lower() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading threat list {THREAT_FILE}: {e}")
        return []

def write_threat_list(wallets):
    unique_wallets = sorted(list(set(w.lower() for w in wallets if w)))
    try:
        with open(THREAT_FILE, 'w') as f:
            for wallet in unique_wallets: f.write(f"{wallet}\n")
    except Exception as e:
        print(f"Error writing threat list {THREAT_FILE}: {e}")

def append_to_csv(filepath, transaction_dict, fieldnames):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    try:
        # Use the global CSV_FIELDNAMES
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            # Write header only if file is new or empty
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader()
            # Ensure all keys exist, defaulting to empty if not present
            row_to_write = {key: transaction_dict.get(key, None) for key in fieldnames}
            writer.writerow(row_to_write)
    except (IOError, OSError, csv.Error, Exception) as e:
        print(f"Error appending to CSV {filepath}: {e}")

def update_csv_status(filepath, tx_hash, new_status, fieldnames):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        # Use global fieldnames
        expected_fieldnames = fieldnames
        try:
            # Read all relevant types as nullable to preserve data
            df = pd.read_csv(filepath, dtype={
                'is_fraud': 'Int64', 
                'ml_fraud_probability': 'float64'
            })
        except (pd.errors.EmptyDataError, FileNotFoundError):
            df = pd.DataFrame(columns=expected_fieldnames)
            if not os.path.exists(filepath):
                 df.to_csv(filepath, index=False)
                 print(f"Created empty file with headers: {filepath}")
            return False

        if tx_hash not in df['tx_hash'].values: return False

        df.loc[df['tx_hash'] == tx_hash, 'final_status'] = new_status
        df.to_csv(filepath, index=False)

        updated_row_series = df[df['tx_hash'] == tx_hash].iloc[0]
        # Use fillna appropriately
        updated_row = updated_row_series.fillna({
             'value_eth': 0.0, 'gas_price': 0.0, 'final_score': 0.0,
             'agent_1_score': 0.0, 'agent_2_score': 0.0, 'agent_3_score': 0.0,
             'ml_fraud_probability': 0.0,
             'reasons': ''
             # Do not fillna 'is_fraud' here, keep it as loaded (could be NA)
        }).to_dict()

        # Convert numeric types
        for key in ['value_eth', 'gas_price', 'final_score', 'agent_1_score', 'agent_2_score', 'agent_3_score', 'ml_fraud_probability']:
             try: updated_row[key] = float(updated_row[key])
             except (ValueError, TypeError): updated_row[key] = 0.0
        updated_row['reasons'] = str(updated_row.get('reasons', ''))
        
        # Convert is_fraud NA to None for JSON compatibility
        if pd.isna(updated_row.get('is_fraud')):
             updated_row['is_fraud'] = None
        elif 'is_fraud' in updated_row:
             updated_row['is_fraud'] = int(updated_row['is_fraud']) # Ensure it's int if not None

        return updated_row
    except Exception as e:
        print(f"Error updating CSV status for {tx_hash}: {e}")
        return False


# --- Transaction Processing ---
def process_incoming_tx(tx_hash):
    global w3
    if not w3 or not w3.is_connected():
        print("[Alchemy Listener] Web3 not connected. Skipping.")
        return

    try:
        tx_hash_hex = tx_hash.hex()
        print(f"[Alchemy Listener] Processing Tx Hash: {tx_hash_hex[:10]}...")
        tx_data = w3.eth.get_transaction(tx_hash)
        if not tx_data:
            print(f"[Alchemy Listener] Tx {tx_hash_hex[:10]}... data not found (likely dropped).")
            return

        adapted_tx = {
            "tx_hash": tx_data.hash.hex(),
            "from_address": tx_data.get('from', '').lower() if tx_data.get('from') else '',
            "to_address": tx_data.get('to', '').lower() if tx_data.get('to') else '',
            "value_eth": float(Web3.from_wei(tx_data.get('value', 0), 'ether')),
            "gas_price": float(Web3.from_wei(tx_data.get('gasPrice', 0), 'gwei')),
            "timestamp": datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'), # Fixed deprecation
            "gas_limit": tx_data.get('gas', 0),
            "block_number": tx_data.get('blockNumber', None),
            # is_fraud & ml_fraud_probability will NOT be present here
        }

        # This now calls the pipeline with the LIVE Etherscan agent
        processed_tx = process_transaction_pipeline(adapted_tx.copy())
        status = processed_tx.get('final_status')
        score = processed_tx.get('final_score', 0)

        print(f"  -> Scanned. Status: {status}, Score: {score:.1f}. Emitting to live feed.")
        socketio.emit('new_scanned_transaction', processed_tx)

        # Append ALL transactions to the CSV using global fieldnames
        append_to_csv(OUTPUT_FILE, processed_tx, CSV_FIELDNAMES)
        print(f"  -> Saved {status} transaction to CSV.")

        # Optional: Emit specific event ONLY for flagged/denied
        if status in ['FLAG_FOR_REVIEW', 'DENY']:
            print(f"  -> FLAGGED ({status}). Emitting dedicated flagged event.")
            socketio.emit('new_flagged_transaction', processed_tx)

    except Exception as e:
        if "not found" in str(e).lower() or isinstance(e, ValueError) and "Transaction" in str(e) and "not found" in str(e):
            print(f"[Alchemy Listener] Tx {tx_hash_hex[:10]}... likely dropped or already mined before fetch.")
        else:
            print(f"[Alchemy Listener] Error processing tx {tx_hash_hex[:10]}...: {e}")


# --- Alchemy Listener Loop ---
def alchemy_listener_loop():
    global alchemy_listener_running, w3
    print("[Alchemy Listener] Starting listener loop...")
    while alchemy_listener_running:
        try:
            w3 = Web3(HTTPProvider(ALCHEMY_HTTP_URL))
            if not w3.is_connected():
                print("[Alchemy Listener] Failed to connect. Retrying in 10s...")
                time.sleep(10)
                continue
            print("[Alchemy Listener] Connected to Alchemy via HTTP.")

            pending_tx_filter = w3.eth.filter('pending')
            print("[Alchemy Listener] Subscribed to pending transactions.")

            while alchemy_listener_running and w3.is_connected():
                try:
                    new_tx_hashes = pending_tx_filter.get_new_entries()
                    if new_tx_hashes:
                        print(f"[Alchemy Listener] Received {len(new_tx_hashes)} new transaction hashes.")
                        for tx_hash in new_tx_hashes:
                            process_incoming_tx(tx_hash)
                            time.sleep(0.05) 

                    time.sleep(2) 

                except Exception as loop_err:
                    print(f"[Alchemy Listener] Error in polling loop: {loop_err}")
                    break
            
            print("[Alchemy Listener] Disconnected or polling loop error. Will attempt to reconnect...")
            w3 = None
            time.sleep(5)

        except Exception as e:
            print(f"Major error in listener setup: {e}")
            w3 = None
            print("[Alchemy Listener] Retrying connection in 15 seconds...")
            time.sleep(15)

    print("[Alchemy Listener] Listener loop stopped.")


# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    global alchemy_thread, alchemy_listener_running
    print('Client connected:', request.sid)
    if not alchemy_listener_running and ALCHEMY_HTTP_URL != "YOUR_ALCHEMY_HTTP_URL":
        print("Starting Alchemy listener task.")
        alchemy_listener_running = True
        alchemy_thread = threading.Thread(target=alchemy_listener_loop, daemon=True)
        alchemy_thread.start()
    
    # Emit current status on connect
    db_ready = os.path.exists(DB_FILE)
    threat_ready = os.path.exists(THREAT_FILE)
    output_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 50 
    emit('status_update', {
        'websocket_connected': True,
        'listener_active': alchemy_listener_running,
        'database_ready': db_ready, # Still indicates if setup was run
        'threat_list_available': threat_ready,
        'has_flagged_data': output_exists
    })


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)


# --- API Endpoints ---
@app.route('/api/status', methods=['GET'])
def get_status():
    db_file_exists = os.path.exists(DB_FILE)
    threat_file_exists = os.path.exists(THREAT_FILE)
    output_file_exists = os.path.exists(OUTPUT_FILE)
    has_data = output_file_exists and os.path.getsize(OUTPUT_FILE) > 50

    return jsonify({
        'database_ready': db_file_exists, # True if setup.bat was run
        'simulation_running': alchemy_listener_running,
        'threat_list_available': threat_file_exists,
        'has_flagged_data': has_data, 
        'listener_active': alchemy_listener_running
    })


@app.route('/api/setup', methods=['POST'])
def run_setup():
    try:
        # Script path relative to PROJECT_ROOT
        setup_script_path = os.path.join(PROJECT_ROOT, 'backend', 'scripts', 'initialize_and_train.py')
        result = subprocess.run([PYTHON_EXE, setup_script_path], capture_output=True, text=True, check=True)
        print(result.stdout)

        # Ensure the output CSV exists with headers (using global fieldnames)
        if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
            print(f"Ensured output file exists with headers: {OUTPUT_FILE}")

        return jsonify({"message": "Setup script executed successfully. Data files initialized."}), 200
    except subprocess.CalledProcessError as e:
        print(f"Setup Error Output: {e.stderr}")
        return jsonify({"error": f"An error occurred during setup: {e.stderr}"}), 500
    except FileNotFoundError:
        return jsonify({"error": f"'{PYTHON_EXE}' or '{setup_script_path}' not found."}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during setup: {str(e)}"}), 500


@app.route('/api/flagged-transactions', methods=['GET'])
def get_flagged_transactions():
    # Endpoint now returns ALL transactions from the CSV
    if not os.path.exists(OUTPUT_FILE):
        return jsonify([])

    try:
        df = pd.read_csv(OUTPUT_FILE, dtype={
            'is_fraud': 'Int64',
            'ml_fraud_probability': 'float64'
        })
        if df.empty:
            return jsonify([])

        # Fill NaNs appropriately
        df = df.fillna({
            'value_eth': 0.0, 'gas_price': 0.0, 'final_score': 0.0,
            'agent_1_score': 0.0, 'agent_2_score': 0.0, 'agent_3_score': 0.0,
            'ml_fraud_probability': 0.0,
            'reasons': ''
        })

        transactions = df.to_dict('records')

        # Convert types and handle is_fraud NA -> None
        for row in transactions:
            for key in ['value_eth', 'gas_price', 'final_score', 'agent_1_score', 'agent_2_score', 'agent_3_score', 'ml_fraud_probability']:
                try: row[key] = float(row[key])
                except (ValueError, TypeError): row[key] = 0.0
            row['reasons'] = str(row.get('reasons', ''))
            
            if pd.isna(row.get('is_fraud')):
                 row['is_fraud'] = None
            elif 'is_fraud' in row:
                 row['is_fraud'] = int(row['is_fraud']) 

        transactions.sort(key=lambda x: x.get('timestamp', '1970-01-01T00:00:00Z'), reverse=True)
        return jsonify(transactions)

    except pd.errors.EmptyDataError:
        return jsonify([])
    except Exception as e:
        print(f"Error reading transactions CSV: {e}")
        return jsonify({"error": f"Failed to read transaction data: {str(e)}"}), 500

# --- /api/wallet/<address> (REPLACED) ---
@app.route('/api/wallet-profile/<address>', methods=['GET'])
def get_wallet_profile_api(address):
    """
    Fetches a wallet profile using the live Etherscan agent.
    """
    if not address:
        return jsonify({"error": "No address provided"}), 400
    try:
        # Use the agent_2's method, which includes caching
        profile = agent_2.get_wallet_profile_from_etherscan(address)
        
        if profile:
            # Calculate age_days for a nice API response
            age_seconds = time.time() - profile['first_tx_timestamp']
            profile['age_days'] = round(age_seconds / 86400, 2)
            # Add a "human" name
            profile['profile_name'] = f"Live Profile ({address[:6]}...{address[-4:]})"
            # Add fields to match the old DB structure for the UI
            profile['wallet_address'] = address
            profile['historical_flagged_tx'] = "N/A (Live)" 
            return jsonify(profile)
        else:
            # This could be an API error or a truly new/invalid address
            return jsonify({"error": "Failed to fetch profile from Etherscan or address is invalid."}), 404
            
    except Exception as e:
        print(f"Error in wallet profile API: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500


# --- /api/review/<tx_hash> (Uses global CSV_FIELDNAMES now) ---
@app.route('/api/review/<tx_hash>', methods=['POST'])
def submit_review(tx_hash):
    data = request.get_json()
    new_status = data.get('status')
    if not new_status or new_status not in ['APPROVE', 'DENY', 'FLAG_FOR_REVIEW']:
        return jsonify({"error": "Invalid status provided."}), 400

    if not os.path.exists(OUTPUT_FILE):
        return jsonify({"error": "Transaction file not found."}), 404

    # Use global fieldnames for the update function
    updated_transaction = update_csv_status(OUTPUT_FILE, tx_hash, new_status, CSV_FIELDNAMES)

    if updated_transaction:
        socketio.emit('transaction_update', updated_transaction)
        print(f"Review submitted: Tx {tx_hash[:10]}... updated to {new_status}")
        return jsonify({"message": f"Transaction {tx_hash} status updated to {new_status}"}), 200
    else:
        try:
            if os.path.exists(OUTPUT_FILE):
                df = pd.read_csv(OUTPUT_FILE)
                if tx_hash not in df['tx_hash'].values:
                    return jsonify({"error": f"Transaction hash {tx_hash} not found in the file."}), 404
            else: 
                 return jsonify({"error": "Transaction file disappeared unexpectedly."}), 500
        except Exception:
            pass 
        return jsonify({"error": "Failed to update transaction status. Check logs."}), 500

# --- /api/threats endpoints ---
@app.route('/api/threats', methods=['GET'])
def get_threats():
    try:
        return jsonify(read_threat_list())
    except Exception as e:
        return jsonify({"error": f"Failed to read threat list: {str(e)}"}), 500

@app.route('/api/threats', methods=['POST'])
def add_threat():
    data = request.get_json()
    new_wallet = data.get('wallet_address', '').strip().lower()
    if not new_wallet.startswith('0x') or len(new_wallet) != 42:
        return jsonify({"error": "Invalid wallet address format."}), 400
    try:
        current_wallets = read_threat_list()
        if new_wallet in current_wallets:
            return jsonify({"message": "Wallet address already in threat list."}), 200

        current_wallets.append(new_wallet)
        write_threat_list(current_wallets)
        updated_list = read_threat_list()
        socketio.emit('threat_list_updated', updated_list)
        print(f"Threat added: {new_wallet}")
        return jsonify({"message": f"Wallet address {new_wallet} added to threat list."}), 201
    except Exception as e:
        return jsonify({"error": f"Failed to add threat: {str(e)}"}), 500

@app.route('/api/threats', methods=['DELETE'])
def remove_threat():
    data = request.get_json()
    wallet_to_remove = data.get('wallet_address', '').strip().lower()
    if not wallet_to_remove:
         return jsonify({"error": "Wallet address required."}), 400

    try:
        current_wallets = read_threat_list()
        if wallet_to_remove not in current_wallets:
            return jsonify({"error": "Wallet address not found in threat list."}), 404

        updated_wallets = [w for w in current_wallets if w != wallet_to_remove]
        write_threat_list(updated_wallets)
        socketio.emit('threat_list_updated', updated_wallets)
        print(f"Threat removed: {wallet_to_remove}")
        return jsonify({"message": f"Wallet address {wallet_to_remove} removed from threat list."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to remove threat: {str(e)}"}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # Use the absolute paths defined at the top
    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'output'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)

    # Check if setup has been run
    if not os.path.exists(DB_FILE) or not os.path.exists(THREAT_FILE) or not os.path.exists(MODEL_FILE):
        print("-" * 60)
        print("Warning: One or more data/model files missing.")
        print(f"DB: {DB_FILE} (Exists: {os.path.exists(DB_FILE)})")
        print(f"Threat: {THREAT_FILE} (Exists: {os.path.exists(THREAT_FILE)})")
        print(f"Model: {MODEL_FILE} (Exists: {os.path.exists(MODEL_FILE)})")
        print("Please run the setup script ('setup.bat' or 'python backend/scripts/initialize_and_train.py')")
        print("before starting the backend to ensure proper functionality.")
        print("-" * 60)

    # Ensure output CSV exists with headers (using global fieldnames)
    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
        print(f"Ensured output file exists with headers: {OUTPUT_FILE}")

    print("Starting Flask-SocketIO server...")
    socketio.run(app, debug=True, port=5000, host='0.0.0.0', use_reloader=False)

    alchemy_listener_running = False
    if alchemy_thread and alchemy_thread.is_alive():
        print("Waiting for Alchemy listener thread to stop...")
        alchemy_thread.join(timeout=2)
    print("Backend server stopped.")