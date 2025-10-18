import csv
import json
import os
import subprocess
import sqlite3
import pandas as pd
import time
import random
import threading
from datetime import datetime
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
DB_FILE = 'data/wallet_profiles.db'
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

# --- Helper Functions (No changes here) ---
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
        df = pd.read_csv(filepath)
        if tx_hash not in df['tx_hash'].values: return False
        df.loc[df['tx_hash'] == tx_hash, 'final_status'] = new_status
        df.to_csv(filepath, index=False)
        updated_row = df[df['tx_hash'] == tx_hash].iloc[0].to_dict()
        for key in ['value_eth', 'gas_price', 'final_score', 'agent_1_score', 'agent_2_score', 'agent_3_score']:
            if key in updated_row:
                try: updated_row[key] = float(updated_row[key])
                except (ValueError, TypeError): updated_row[key] = 0.0
        return updated_row
    except (pd.errors.EmptyDataError, FileNotFoundError, Exception) as e:
        print(f"Error updating CSV status: {e}")
        return False

# --- Transaction Processing (No changes here) ---
def process_incoming_tx(tx_hash):
    global w3
    if not w3 or not w3.is_connected():
        print("[Alchemy Listener] Web3 not connected. Skipping.")
        return

    try:
        tx_hash_hex = tx_hash.hex()
        print(f"[Alchemy Listener] Processing Tx Hash: {tx_hash_hex[:10]}...")
        tx_data = w3.eth.get_transaction(tx_hash)
        if not tx_data: return

        adapted_tx = {
            "tx_hash": tx_data.hash.hex(),
            "from_address": tx_data.get('from', '').lower() if tx_data.get('from') else '',
            "to_address": tx_data.get('to', '').lower() if tx_data.get('to') else '',
            "value_eth": float(Web3.from_wei(tx_data.get('value', 0), 'ether')),
            "gas_price": float(Web3.from_wei(tx_data.get('gasPrice', 0), 'gwei')),
            "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            "gas_limit": tx_data.get('gas', 0),
            "block_number": tx_data.get('blockNumber', None),
        }

        processed_tx = process_transaction_pipeline(adapted_tx.copy())
        status = processed_tx.get('final_status')
        score = processed_tx.get('final_score', 0)

        print(f"  -> Scanned. Status: {status}, Score: {score:.1f}. Emitting to live feed.")
        socketio.emit('new_scanned_transaction', processed_tx)

        if status in ['FLAG_FOR_REVIEW', 'DENY']:
            print(f"  -> FLAGGED. Emitting to flagged list.")
            csv_fieldnames = [
                'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
                'timestamp', 'final_status', 'final_score', 'reasons',
                'agent_1_score', 'agent_2_score', 'agent_3_score'
            ]
            append_to_csv(OUTPUT_FILE, processed_tx, csv_fieldnames)
            socketio.emit('new_flagged_transaction', processed_tx)

    except Exception as e:
        if "not found" in str(e).lower():
            print(f"[Alchemy Listener] Tx {tx_hash_hex[:10]}... dropped or already mined.")
        else:
            print(f"[Alchemy Listener] Error processing tx {tx_hash_hex[:10]}...: {e}")

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
                        print(f"[Alchemy Listener] Received {len(new_tx_hashes)} new transactions.")
                        for tx_hash in new_tx_hashes:
                            # We can process them in parallel with threads for better performance,
                            # but sequentially is safer to start with.
                            process_incoming_tx(tx_hash)
                    
                    # Wait a short duration before polling again
                    time.sleep(2)
                
                except Exception as loop_err:
                    print(f"[Alchemy Listener] Error in polling loop: {loop_err}")
                    # If there's an error, the connection might be dead. Break to reconnect.
                    break
            
            # If the loop breaks, it means we disconnected.
            print("[Alchemy Listener] Disconnected. Will attempt to reconnect...")
            w3 = None
            time.sleep(5) # Wait before reconnecting

        except Exception as e:
            print(f"[Alchemy Listener] Major error in listener setup: {e}")
            w3 = None
            print("[Alchemy Listener] Retrying connection in 15 seconds...")
            time.sleep(15)

    print("[Alchemy Listener] Listener loop stopped.")


# --- SocketIO Events (No changes here) ---
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


# --- API Endpoints (No changes here) ---
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'database_ready': os.path.exists(DB_FILE),
        'simulation_running': alchemy_listener_running,
        'threat_list_available': os.path.exists(THREAT_FILE),
        'has_flagged_data': os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 150,
        'listener_active': alchemy_listener_running
    })

@app.route('/api/setup', methods=['POST'])
def run_setup():
    try:
        result = subprocess.run([PYTHON_EXE, '0_setup_database.py'], capture_output=True, text=True, check=True)
        print(result.stdout)
        csv_fieldnames = [
            'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
            'timestamp', 'final_status', 'final_score', 'reasons',
            'agent_1_score', 'agent_2_score', 'agent_3_score'
        ]
        if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
            append_to_csv(OUTPUT_FILE, {}, csv_fieldnames)
        return jsonify({"message": "Database and models set up successfully"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"An error occurred during setup: {e.stderr}"}), 500
    except FileNotFoundError:
        return jsonify({"error": "Python executable not found."}), 500

@app.route('/api/flagged-transactions', methods=['GET'])
def get_flagged_transactions():
    if not os.path.exists(OUTPUT_FILE): return jsonify([])
    try:
        df = pd.read_csv(OUTPUT_FILE)
        if df.empty: return jsonify([])
        df = df.fillna(0)
        transactions = df.to_dict('records')
        for row in transactions:
            for key in ['value_eth', 'gas_price', 'final_score', 'agent_1_score', 'agent_2_score', 'agent_3_score']:
                if key in row:
                    try: row[key] = float(row[key])
                    except (ValueError, TypeError): row[key] = 0.0
            row['reasons'] = str(row.get('reasons', ''))
        transactions.sort(key=lambda x: x.get('timestamp', '1970-01-01T00:00:00Z'), reverse=True)
        return jsonify(transactions)
    except (pd.errors.EmptyDataError, Exception) as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/wallet/<address>', methods=['GET'])
def get_wallet_profile(address):
    if not os.path.exists(DB_FILE): return jsonify({"error": "Database not found."}), 404
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM wallet_profiles WHERE wallet_address = ?", (address.lower(),))
        profile = cursor.fetchone()
        conn.close()
        if profile: return jsonify(dict(profile))
        return jsonify({"message": "Wallet profile not found"}), 404
    except sqlite3.Error as e: return jsonify({"error": f"Database error: {e}"}), 500

@app.route('/api/review/<tx_hash>', methods=['POST'])
def submit_review(tx_hash):
    data = request.get_json()
    new_status = data.get('status')
    if not new_status or new_status not in ['APPROVE', 'DENY', 'FLAG_FOR_REVIEW']: return jsonify({"error": "Invalid status."}), 400
    if not os.path.exists(OUTPUT_FILE): return jsonify({"error": "File not found."}), 404
    csv_fieldnames = [
        'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
        'timestamp', 'final_status', 'final_score', 'reasons',
        'agent_1_score', 'agent_2_score', 'agent_3_score'
    ]
    updated_transaction = update_csv_status(OUTPUT_FILE, tx_hash, new_status, csv_fieldnames)
    if updated_transaction:
        socketio.emit('transaction_update', updated_transaction)
        return jsonify({"message": f"Transaction {tx_hash} updated to {new_status}"}), 200
    return jsonify({"error": "Failed to update transaction."}), 500

@app.route('/api/threats', methods=['GET'])
def get_threats():
    try: return jsonify(read_threat_list())
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/threats', methods=['POST'])
def add_threat():
    data = request.get_json()
    new_wallet = data.get('wallet_address', '').strip().lower()
    if not new_wallet.startswith('0x') or len(new_wallet) != 42: return jsonify({"error": "Invalid wallet format"}), 400
    try:
        current_wallets = read_threat_list()
        if new_wallet in current_wallets: return jsonify({"message": "Wallet already exists"}), 200
        current_wallets.append(new_wallet)
        write_threat_list(current_wallets)
        socketio.emit('threat_list_updated', read_threat_list())
        return jsonify({"message": f"Wallet {new_wallet} added."}), 201
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/threats', methods=['DELETE'])
def remove_threat():
    data = request.get_json()
    wallet_to_remove = data.get('wallet_address', '').strip().lower()
    try:
        current_wallets = read_threat_list()
        if wallet_to_remove not in current_wallets: return jsonify({"error": "Wallet not found"}), 404
        updated_wallets = [w for w in current_wallets if w != wallet_to_remove]
        write_threat_list(updated_wallets)
        socketio.emit('threat_list_updated', updated_wallets)
        return jsonify({"message": f"Wallet {wallet_to_remove} removed."}), 200
    except Exception as e: return jsonify({"error": str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    if not os.path.exists(DB_FILE) or not os.path.exists(THREAT_FILE):
        print("-" * 60, "\nWarning: Data files missing. Run setup.\n", "-" * 60)

    print("Starting Flask-SocketIO server...")
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')

    alchemy_listener_running = False
    if alchemy_thread:
        alchemy_thread.join(timeout=2)