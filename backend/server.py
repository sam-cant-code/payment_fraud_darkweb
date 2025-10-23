import csv
import json
import os
import subprocess
import sqlite3
import pandas as pd
import time
import random
import threading
import sys  # <-- Import sys to read command-line args
from datetime import datetime, UTC
from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from web3 import Web3
from web3.providers import HTTPProvider
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# --- MODIFIED IMPORT ---
# This now only imports the functions that exist in your new agents.py
try:
    from app.agents import process_transaction_pipeline, get_system_thresholds
except ImportError:
    print("="*80)
    print("FATAL ERROR: Could not import from 'app.agents'.")
    print("Please ensure your new 'agents.py' file (V3.4) is saved at 'backend/app/agents.py'")
    print("="*80)
    sys.exit(1)


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/socket.io/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Path Definitions (Relative to this server.py file) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'output', 'all_processed_transactions.csv')
DB_FILE = os.path.join(PROJECT_ROOT, 'data', 'wallet_profiles.db') # Now used for wallet API
THREAT_FILE = os.path.join(PROJECT_ROOT, 'data', 'dark_web_wallets.txt')
PYTHON_EXE = 'python' # Assumes 'python' is in your PATH

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

# This model timestamp will be passed to process_transaction_pipeline
# It's set from the command-line argument when the server starts
SELECTED_MODEL_TIMESTAMP = None
if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
    SELECTED_MODEL_TIMESTAMP = sys.argv[1]
    print(f"[Server] Using specific model: {SELECTED_MODEL_TIMESTAMP}")
else:
    print("[Server] Using 'latest' model (default)")


# --- CSV Fieldnames (including is_fraud) ---
CSV_FIELDNAMES = [
    'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
    'timestamp', 'final_status', 'final_score', 'reasons',
    'agent_1_score', 'agent_2_score', 'agent_3_score',
    'ml_fraud_probability',
    'is_fraud'
]


# --- Helper Functions (Unchanged) ---
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader()
            row_to_write = {key: transaction_dict.get(key, None) for key in fieldnames}
            writer.writerow(row_to_write)
    except (IOError, OSError, csv.Error, Exception) as e:
        print(f"Error appending to CSV {filepath}: {e}")

def update_csv_status(filepath, tx_hash, new_status, fieldnames):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        try:
            df = pd.read_csv(filepath, dtype={
                'is_fraud': 'Int64', 
                'ml_fraud_probability': 'float64'
            })
        except (pd.errors.EmptyDataError, FileNotFoundError):
            df = pd.DataFrame(columns=fieldnames)
            if not os.path.exists(filepath):
                df.to_csv(filepath, index=False)
                print(f"Created empty file with headers: {filepath}")
            return False

        if tx_hash not in df['tx_hash'].values: return False

        df.loc[df['tx_hash'] == tx_hash, 'final_status'] = new_status
        df.to_csv(filepath, index=False)

        updated_row_series = df[df['tx_hash'] == tx_hash].iloc[0]
        updated_row = updated_row_series.fillna({
            'value_eth': 0.0, 'gas_price': 0.0, 'final_score': 0.0,
            'agent_1_score': 0.0, 'agent_2_score': 0.0, 'agent_3_score': 0.0,
            'ml_fraud_probability': 0.0, 'reasons': ''
        }).to_dict()

        for key in ['value_eth', 'gas_price', 'final_score', 'agent_1_score', 'agent_2_score', 'agent_3_score', 'ml_fraud_probability']:
            try: updated_row[key] = float(updated_row[key])
            except (ValueError, TypeError): updated_row[key] = 0.0
        row['reasons'] = str(row.get('reasons', ''))
        
        if pd.isna(updated_row.get('is_fraud')):
            updated_row['is_fraud'] = None
        elif 'is_fraud' in updated_row:
            updated_row['is_fraud'] = int(updated_row['is_fraud'])

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
            "timestamp": datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "gas_limit": tx_data.get('gas', 0),
            "block_number": tx_data.get('blockNumber', None),
        }

        # --- MODIFIED CALL ---
        # This now calls the pipeline and passes the specific model timestamp
        # that was chosen when the server started.
        processed_tx = process_transaction_pipeline(
            adapted_tx.copy(), 
            model_timestamp=SELECTED_MODEL_TIMESTAMP
        )
        status = processed_tx.get('final_status')
        score = processed_tx.get('final_score', 0)

        print(f"  -> Scanned. Status: {status}, Score: {score:.1f}. Emitting to live feed.")
        socketio.emit('new_scanned_transaction', processed_tx)

        # Append ALL transactions to the CSV
        append_to_csv(OUTPUT_FILE, processed_tx, CSV_FIELDNAMES)
        print(f"  -> Saved {status} transaction to CSV.")

        if status in ['FLAG_FOR_REVIEW', 'DENY']:
            print(f"  -> FLAGGED ({status}). Emitting dedicated flagged event.")
            socketio.emit('new_flagged_transaction', processed_tx)

    except Exception as e:
        if "not found" in str(e).lower() or isinstance(e, ValueError) and "Transaction" in str(e) and "not found" in str(e):
            print(f"[Alchemy Listener] Tx {tx_hash_hex[:10]}... likely dropped or already mined before fetch.")
        else:
            print(f"[Alchemy Listener] Error processing tx {tx_hash_hex[:10]}...: {e}")


# --- Alchemy Listener Loop (Unchanged) ---
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


# --- SocketIO Events (Unchanged) ---
@socketio.on('connect')
def handle_connect():
    global alchemy_thread, alchemy_listener_running
    print('Client connected:', request.sid)
    if not alchemy_listener_running and ALCHEMY_HTTP_URL != "YOUR_ALCHEMY_HTTP_URL":
        print("Starting Alchemy listener task.")
        alchemy_listener_running = True
        alchemy_thread = threading.Thread(target=alchemy_listener_loop, daemon=True)
        alchemy_thread.start()
    
    db_ready = os.path.exists(DB_FILE)
    threat_ready = os.path.exists(THREAT_FILE)
    output_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 50 
    emit('status_update', {
        'websocket_connected': True,
        'listener_active': alchemy_listener_running,
        'database_ready': db_ready,
        'threat_list_available': threat_ready,
        'has_flagged_data': output_exists
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)


# --- API Endpoints ---
@app.route('/api/status', methods=['GET'])
def get_status():
    """
    This endpoint now also returns the thresholds from the new agent system.
    """
    db_file_exists = os.path.exists(DB_FILE)
    threat_file_exists = os.path.exists(THREAT_FILE)
    output_file_exists = os.path.exists(OUTPUT_FILE)
    has_data = output_file_exists and os.path.getsize(OUTPUT_FILE) > 50
    
    try:
        thresholds = get_system_thresholds()
    except Exception:
        thresholds = {"error": "Could not load thresholds from agents.py"}

    return jsonify({
        'database_ready': db_file_exists,
        'simulation_running': alchemy_listener_running,
        'threat_list_available': threat_file_exists,
        'has_flagged_data': has_data, 
        'listener_active': alchemy_listener_running,
        'thresholds': thresholds, # <-- Added thresholds
        'model_in_use': SELECTED_MODEL_TIMESTAMP or 'latest' # <-- Added model info
    })


@app.route('/api/setup', methods=['POST'])
def run_setup():
    try:
        setup_script_path = os.path.join(PROJECT_ROOT, 'backend', 'scripts', 'initialize_and_train.py')
        print(f"Running setup script: {setup_script_path}")
        result = subprocess.run([PYTHON_EXE, setup_script_path], capture_output=True, text=True, check=True)
        print(result.stdout)

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
    if not os.path.exists(OUTPUT_FILE):
        return jsonify([])
    try:
        df = pd.read_csv(OUTPUT_FILE, dtype={
            'is_fraud': 'Int64',
            'ml_fraud_probability': 'float64'
        })
        if df.empty:
            return jsonify([])

        df = df.fillna({
            'value_eth': 0.0, 'gas_price': 0.0, 'final_score': 0.0,
            'agent_1_score': 0.0, 'agent_2_score': 0.0, 'agent_3_score': 0.0,
            'ml_fraud_probability': 0.0, 'reasons': ''
        })
        transactions = df.to_dict('records')

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
    --- MODIFIED ENDPOINT ---
    Fetches a wallet's historical profile from the 'wallet_profiles.db'
    This uses the same data source as the new ML model.
    """
    if not address:
        return jsonify({"error": "No address provided"}), 400
    
    if not os.path.exists(DB_FILE):
        return jsonify({"error": "Wallet profiles database not found. Please run setup."}), 404
        
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM wallet_profiles WHERE wallet_address = ?", (address.lower(),))
        profile_data = cursor.fetchone()
        conn.close()
        
        if profile_data:
            profile = dict(profile_data) # Convert sqlite3.Row to dict
            
            # Add calculated fields for the UI
            profile['profile_name'] = f"DB Profile ({profile['wallet_address'][:6]}...{profile['wallet_address'][-4:]})"
            
            try:
                first_seen_dt = datetime.fromisoformat(profile['first_seen'].replace('Z', ''))
                age_seconds = (datetime.now() - first_seen_dt).total_seconds()
                profile['age_days'] = round(age_seconds / 86400, 2)
            except:
                profile['age_days'] = "N/A"
                
            return jsonify(profile)
        else:
            # Wallet is not in the DB (it's new or has no history)
            return jsonify({
                "wallet_address": address,
                "profile_name": f"New Wallet ({address[:6]}...{address[-4:]})",
                "avg_sent_value": 0, "tx_count_sent": 0, "unique_recipients": 0,
                "avg_time_between_tx": 0, "min_time_between_tx": 0,
                "tx_count_received": 0, "total_received": 0, "unique_senders": 0,
                "age_days": 0, "first_seen": "N/A", "last_seen": "N/A"
            }), 200
            
    except Exception as e:
        print(f"Error in wallet profile API: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500


# --- /api/review/<tx_hash> (Unchanged) ---
@app.route('/api/review/<tx_hash>', methods=['POST'])
def submit_review(tx_hash):
    data = request.get_json()
    new_status = data.get('status')
    if not new_status or new_status not in ['APPROVE', 'DENY', 'FLAG_FOR_REVIEW']:
        return jsonify({"error": "Invalid status provided."}), 400

    if not os.path.exists(OUTPUT_FILE):
        return jsonify({"error": "Transaction file not found."}), 404

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

# --- /api/threats endpoints (Unchanged) ---
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
    # Create required directories
    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'output'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)

    # Check if setup has been run
    # NOTE: This no longer checks for 'behavior_model.pkl'
    if not os.path.exists(DB_FILE) or not os.path.exists(THREAT_FILE):
        print("-" * 60)
        print("Warning: One or more data/model files missing.")
        print(f"DB: {DB_FILE} (Exists: {os.path.exists(DB_FILE)})")
        print(f"Threat: {THREAT_FILE} (Exists: {os.path.exists(THREAT_FILE)})")
        print("Please run the setup script ('setup.bat' or 'python backend/scripts/initialize_and_train.py')")
        print("before starting the backend to ensure proper functionality.")
        print("-" * 60)
    else:
        print("✓ Database and threat list found.")

    # Ensure output CSV exists with headers
    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
        print(f"Ensured output file exists with headers: {OUTPUT_FILE}")

    # --- MODIFIED SERVER START ---
    # We use 'debug=True' for auto-reloading, but 'use_reloader=False'
    # would disable it. Flask's reloader is better.
    # We will pass 'debug=True' and let Flask handle reloading.
    # The 'os.chdir' was removed from my previous file, so this won't crash.
    print("Starting Flask-SocketIO server...")
    port = int(os.environ.get('PORT', 5000))
    # 'debug=True' enables the reloader.
    # 'allow_unsafe_werkzeug=True' is sometimes needed for newer Flask versions
    # when running in debug mode from a script.
    socketio.run(app, debug=True, port=port, host='0.0.0.0', allow_unsafe_werkzeug=True)

    # This code below will only run *after* the server is stopped (e.g., Ctrl+C)
    alchemy_listener_running = False
    if alchemy_thread and alchemy_thread.is_alive():
        print("Waiting for Alchemy listener thread to stop...")
        alchemy_thread.join(timeout=2)
    print("Backend server stopped.")