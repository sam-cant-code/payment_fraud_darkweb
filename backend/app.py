import csv
import json
import os
import subprocess
import sqlite3
import pandas as pd
import time
import random
import threading
from datetime import datetime, UTC # Added UTC
from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from web3 import Web3
# CORRECTED IMPORT for web3.py v6+
from web3.providers import HTTPProvider
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()

# Import agent logic
from agents import process_transaction_pipeline, load_dark_web_wallets

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/socket.io/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

OUTPUT_FILE = 'output/flagged_transactions.csv'
DB_FILE = 'data/wallet_profiles.db' # Path used by agents and setup
THREAT_FILE = 'data/dark_web_wallets.txt'
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

# --- Helper Functions ---
def read_threat_list():
    if not os.path.exists(THREAT_FILE): return []
    with open(THREAT_FILE, 'r') as f: return [line.strip().lower() for line in f if line.strip()]

def write_threat_list(wallets):
    unique_wallets = sorted(list(set(w.lower() for w in wallets if w)))
    with open(THREAT_FILE, 'w') as f:
        for wallet in unique_wallets: f.write(f"{wallet}\n")

def append_to_csv(filepath, transaction_dict, fieldnames):
    file_exists = os.path.isfile(filepath)
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists or os.path.getsize(filepath) == 0: writer.writeheader()
            writer.writerow(transaction_dict)
    except (IOError, Exception) as e: print(f"Error appending to CSV {filepath}: {e}")

def update_csv_status(filepath, tx_hash, new_status, fieldnames):
    try:
        # Define fieldnames explicitly in case file is empty or missing headers
        expected_fieldnames = [
            'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
            'timestamp', 'final_status', 'final_score', 'reasons',
            'agent_1_score', 'agent_2_score', 'agent_3_score'
        ]
        try:
            df = pd.read_csv(filepath)
        except (pd.errors.EmptyDataError, FileNotFoundError):
             # If file is empty/missing, create an empty DataFrame with headers
            df = pd.DataFrame(columns=expected_fieldnames)
            if not os.path.exists(filepath): # Create file if it doesn't exist
                 df.to_csv(filepath, index=False)
                 print(f"Created empty file with headers: {filepath}")
            return False # Cannot update if file was just created or empty

        if tx_hash not in df['tx_hash'].values: return False

        df.loc[df['tx_hash'] == tx_hash, 'final_status'] = new_status
        df.to_csv(filepath, index=False) # Overwrite the file with updated data

        # Reload the specific row to return it
        updated_row_series = df[df['tx_hash'] == tx_hash].iloc[0]
        updated_row = updated_row_series.fillna(0).to_dict() # Fill NaN with 0 before converting

        # Ensure numeric types are correct
        for key in ['value_eth', 'gas_price', 'final_score', 'agent_1_score', 'agent_2_score', 'agent_3_score']:
            if key in updated_row:
                try: updated_row[key] = float(updated_row[key])
                except (ValueError, TypeError): updated_row[key] = 0.0 # Default to 0.0 on error
        updated_row['reasons'] = str(updated_row.get('reasons', '')) # Ensure reasons is a string

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
            # Use timezone-aware datetime
            "timestamp": datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'), # <-- FIXED DEPRECATION
            "gas_limit": tx_data.get('gas', 0),
            "block_number": tx_data.get('blockNumber', None),
        }

        processed_tx = process_transaction_pipeline(adapted_tx.copy())
        status = processed_tx.get('final_status')
        score = processed_tx.get('final_score', 0)

        print(f"  -> Scanned. Status: {status}, Score: {score:.1f}. Emitting to live feed.")
        # Emit ALL scanned transactions (this event is now used by frontend)
        socketio.emit('new_scanned_transaction', processed_tx)

        # Always append ALL transactions to the CSV for persistence/initial load
        csv_fieldnames = [
            'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
            'timestamp', 'final_status', 'final_score', 'reasons',
            'agent_1_score', 'agent_2_score', 'agent_3_score' # Ensure agent_3_score is here
        ]
        append_to_csv(OUTPUT_FILE, processed_tx, csv_fieldnames)
        print(f"  -> Saved {status} transaction to CSV.") # Log saving

        # Optionally, still emit a specific event ONLY for flagged/denied if needed elsewhere
        if status in ['FLAG_FOR_REVIEW', 'DENY']:
            print(f"  -> FLAGGED ({status}). Emitting dedicated flagged event.")
            socketio.emit('new_flagged_transaction', processed_tx)

    except Exception as e:
        # Handle cases where tx might disappear between filter and get_transaction
        if "not found" in str(e).lower() or isinstance(e, ValueError) and "Transaction" in str(e) and "not found" in str(e):
            print(f"[Alchemy Listener] Tx {tx_hash_hex[:10]}... likely dropped or already mined before fetch.")
        else:
            print(f"[Alchemy Listener] Error processing tx {tx_hash_hex[:10]}...: {e}")
            # Consider more detailed error logging here if needed


# --- REFACTORED Alchemy HTTP Listener Loop (Polling) ---
def alchemy_listener_loop():
    global alchemy_listener_running, w3
    print("[Alchemy Listener] Starting listener loop...")
    while alchemy_listener_running:
        try:
            # Connect to Alchemy via HTTP
            w3 = Web3(HTTPProvider(ALCHEMY_HTTP_URL))
            if not w3.is_connected():
                print("[Alchemy Listener] Failed to connect. Retrying in 10s...")
                time.sleep(10)
                continue
            print("[Alchemy Listener] Connected to Alchemy via HTTP.")

            # Create a filter for pending transactions
            pending_tx_filter = w3.eth.filter('pending')
            print("[Alchemy Listener] Subscribed to pending transactions.")

            # Poll the filter for new entries
            while alchemy_listener_running and w3.is_connected():
                try:
                    new_tx_hashes = pending_tx_filter.get_new_entries()
                    if new_tx_hashes:
                        print(f"[Alchemy Listener] Received {len(new_tx_hashes)} new transaction hashes.")
                        for tx_hash in new_tx_hashes:
                            # Process transactions sequentially for simplicity
                            process_incoming_tx(tx_hash)
                            time.sleep(0.05) # Small delay to avoid overwhelming API/CPU if many tx come at once

                    # Wait a short duration before polling again
                    time.sleep(2) # Poll every 2 seconds

                except Exception as loop_err:
                    print(f"[Alchemy Listener] Error in polling loop: {loop_err}")
                    # If there's an error, the connection might be dead. Break to reconnect.
                    break

            # If the loop breaks, it means we disconnected.
            print("[Alchemy Listener] Disconnected or polling loop error. Will attempt to reconnect...")
            w3 = None
            time.sleep(5) # Wait before reconnecting

        except Exception as e:
            print(f"[Alchemy Listener] Major error in listener setup: {e}")
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
    emit('status_update', {'websocket_connected': True, 'simulation_running': alchemy_listener_running})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)
    # Note: Listener thread continues running even if clients disconnect,
    # as it's managed by the 'alchemy_listener_running' flag, which isn't reset here.


# --- API Endpoints ---
@app.route('/api/status', methods=['GET'])
def get_status():
    db_file_exists = os.path.exists('data/wallet_profiles.db') # Check specific DB file used
    threat_file_exists = os.path.exists(THREAT_FILE)
    output_file_exists = os.path.exists(OUTPUT_FILE)
    has_data = output_file_exists and os.path.getsize(OUTPUT_FILE) > 50 # Check if CSV has more than just headers potentially
    
    return jsonify({
        'database_ready': db_file_exists, # Check profile DB specifically
        'simulation_running': alchemy_listener_running,
        'threat_list_available': threat_file_exists,
        'has_flagged_data': has_data, # Renamed for clarity, reflects CSV state
        'listener_active': alchemy_listener_running
    })

@app.route('/api/setup', methods=['POST'])
def run_setup():
    try:
        # Run the original setup script which creates mock data, models, and profile DB
        result = subprocess.run([PYTHON_EXE, '0_setup_database.py'], capture_output=True, text=True, check=True)
        print(result.stdout)

        # Ensure the output CSV exists with headers even if setup doesn't create it
        csv_fieldnames = [
            'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
            'timestamp', 'final_status', 'final_score', 'reasons',
            'agent_1_score', 'agent_2_score', 'agent_3_score'
        ]
        if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writeheader()
            print(f"Ensured output file exists with headers: {OUTPUT_FILE}")

        return jsonify({"message": "Setup script executed successfully. Data files initialized."}), 200
    except subprocess.CalledProcessError as e:
        print(f"Setup Error Output: {e.stderr}")
        return jsonify({"error": f"An error occurred during setup: {e.stderr}"}), 500
    except FileNotFoundError:
        return jsonify({"error": f"'{PYTHON_EXE}' or '0_setup_database.py' not found."}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during setup: {str(e)}"}), 500


@app.route('/api/flagged-transactions', methods=['GET'])
def get_flagged_transactions():
    # This now returns ALL transactions saved in the CSV
    if not os.path.exists(OUTPUT_FILE):
        return jsonify([]) # Return empty list if file doesn't exist

    try:
        df = pd.read_csv(OUTPUT_FILE)
        if df.empty:
            return jsonify([]) # Return empty list if file is empty

        # Fill potential missing numeric values with 0 and string values with ''
        df = df.fillna({
            'value_eth': 0.0, 'gas_price': 0.0, 'final_score': 0.0,
            'agent_1_score': 0.0, 'agent_2_score': 0.0, 'agent_3_score': 0.0,
            'reasons': ''
        })

        transactions = df.to_dict('records')

        # Convert numeric types and ensure reasons is string
        for row in transactions:
            for key in ['value_eth', 'gas_price', 'final_score', 'agent_1_score', 'agent_2_score', 'agent_3_score']:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    row[key] = 0.0 # Default to 0.0 on error
            row['reasons'] = str(row.get('reasons', '')) # Ensure reasons is string

        # Sort by timestamp descending (newest first)
        transactions.sort(key=lambda x: x.get('timestamp', '1970-01-01T00:00:00Z'), reverse=True)
        return jsonify(transactions)

    except pd.errors.EmptyDataError:
        return jsonify([]) # Return empty list if pandas reads an empty file
    except Exception as e:
        print(f"Error reading transactions CSV: {e}")
        return jsonify({"error": f"Failed to read transaction data: {str(e)}"}), 500


@app.route('/api/wallet/<address>', methods=['GET'])
def get_wallet_profile(address):
    # Uses the DB file created by 0_setup_database.py
    profile_db_path = 'data/wallet_profiles.db'
    if not os.path.exists(profile_db_path):
        return jsonify({"error": "Wallet profiles database not found. Run setup."}), 404
    try:
        conn = sqlite3.connect(profile_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM wallet_profiles WHERE wallet_address = ?", (address.lower(),))
        profile = cursor.fetchone()
        conn.close()
        if profile:
            return jsonify(dict(profile))
        else:
            return jsonify({"message": "Wallet profile not found"}), 404
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
         return jsonify({"error": f"Unexpected error fetching profile: {str(e)}"}), 500

@app.route('/api/review/<tx_hash>', methods=['POST'])
def submit_review(tx_hash):
    data = request.get_json()
    new_status = data.get('status')
    if not new_status or new_status not in ['APPROVE', 'DENY', 'FLAG_FOR_REVIEW']:
        return jsonify({"error": "Invalid status provided."}), 400

    if not os.path.exists(OUTPUT_FILE):
        return jsonify({"error": "Transaction file not found."}), 404

    # Fieldnames needed for update function
    csv_fieldnames = [
        'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
        'timestamp', 'final_status', 'final_score', 'reasons',
        'agent_1_score', 'agent_2_score', 'agent_3_score'
    ]

    updated_transaction = update_csv_status(OUTPUT_FILE, tx_hash, new_status, csv_fieldnames)

    if updated_transaction:
        # Emit the update to all connected clients
        socketio.emit('transaction_update', updated_transaction)
        print(f"Review submitted: Tx {tx_hash[:10]}... updated to {new_status}")
        return jsonify({"message": f"Transaction {tx_hash} status updated to {new_status}"}), 200
    else:
        # Check if the transaction hash was simply not found
        try:
            df = pd.read_csv(OUTPUT_FILE)
            if tx_hash not in df['tx_hash'].values:
                return jsonify({"error": f"Transaction hash {tx_hash} not found in the file."}), 404
        except Exception:
            pass # Fall through to generic error if file reading fails
        return jsonify({"error": "Failed to update transaction status. Check logs."}), 500


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
    # Basic validation
    if not new_wallet.startswith('0x') or len(new_wallet) != 42:
        return jsonify({"error": "Invalid wallet address format."}), 400
    try:
        current_wallets = read_threat_list()
        if new_wallet in current_wallets:
            return jsonify({"message": "Wallet address already in threat list."}), 200 # Or 409 Conflict

        current_wallets.append(new_wallet)
        write_threat_list(current_wallets)
        # Reload the list to ensure consistency and emit
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
        # Emit the updated list
        socketio.emit('threat_list_updated', updated_wallets)
        print(f"Threat removed: {wallet_to_remove}")
        return jsonify({"message": f"Wallet address {wallet_to_remove} removed from threat list."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to remove threat: {str(e)}"}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # Ensure necessary directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Check for essential files created by setup
    if not os.path.exists('data/wallet_profiles.db') or not os.path.exists(THREAT_FILE) or not os.path.exists('models/behavior_model.pkl'):
        print("-" * 60)
        print("Warning: One or more data/model files missing.")
        print("Please run the setup script ('setup.bat' or 'python backend/0_setup_database.py')")
        print("before starting the backend to ensure proper functionality.")
        print("-" * 60)

    # Ensure output CSV exists with headers
    csv_fieldnames = [
        'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
        'timestamp', 'final_status', 'final_score', 'reasons',
        'agent_1_score', 'agent_2_score', 'agent_3_score'
    ]
    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
        print(f"Ensured output file exists with headers: {OUTPUT_FILE}")

    print("Starting Flask-SocketIO server...")
    # Use host='0.0.0.0' to make it accessible on the network
    # debug=True enables auto-reloading but might run the listener twice in some setups.
    # Set use_reloader=False if you experience duplicate processing with debug=True.
    socketio.run(app, debug=True, port=5000, host='0.0.0.0', use_reloader=False)

    # Cleanup listener thread if it was started
    alchemy_listener_running = False
    if alchemy_thread and alchemy_thread.is_alive():
        print("Waiting for Alchemy listener thread to stop...")
        alchemy_thread.join(timeout=2)
    print("Backend server stopped.")