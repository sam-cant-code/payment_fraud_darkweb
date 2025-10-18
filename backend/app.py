import csv
import json
import os
import subprocess
import sqlite3
import pandas as pd
import time # Added for sleep in simulation loop
import random # Added for simulating new transactions
from datetime import datetime # Added for review timestamp
from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit # Added SocketIO

# Import agent logic
# Ensure agent_3_network.py exists if you kept that code from before
from agents import process_transaction_pipeline, load_dark_web_wallets

app = Flask(__name__)
# Enable CORS for SocketIO as well
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/socket.io/*": {"origins": "*"}})
# Set async_mode to 'eventlet', recommended for performance
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

OUTPUT_FILE = 'output/flagged_transactions.csv'
DB_FILE = 'data/wallet_profiles.db'
THREAT_FILE = 'data/dark_web_wallets.txt'
PYTHON_EXE = 'python'

# --- Global variable to control the simulation loop ---
simulation_running = False
simulation_task = None

# --- Helper Functions --- (read_threat_list, write_threat_list remain the same)
def read_threat_list():
    """Reads the current threat list."""
    if not os.path.exists(THREAT_FILE):
        return []
    with open(THREAT_FILE, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

def write_threat_list(wallets):
    """Writes the updated threat list."""
    unique_wallets = sorted(list(set(w.lower() for w in wallets if w)))
    with open(THREAT_FILE, 'w') as f:
        for wallet in unique_wallets:
            f.write(f"{wallet}\n")

def generate_simulated_transaction():
    """Generates a single random transaction for the real-time feed."""
    # Use lists obtained during setup or define some defaults
    normal_wallets = [
        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "0xcccccccccccccccccccccccccccccccccccccccc",
        "0xdddddddddddddddddddddddddddddddddddddddd",
        "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    ]
    # You might want to occasionally include known malicious ones from THREAT_FILE too
    threat_list = read_threat_list()
    all_wallets = normal_wallets + threat_list

    if not all_wallets: # Fallback if threat list is empty or fails to load
        all_wallets = normal_wallets

    from_addr = random.choice(all_wallets)
    to_addr = random.choice(all_wallets)
    # Avoid self-sends most of the time
    while to_addr == from_addr and len(all_wallets) > 1:
        to_addr = random.choice(all_wallets)

    # Introduce occasional high-value or potentially risky transactions
    is_potentially_risky = random.random() < 0.1 # 10% chance
    value = round(random.uniform(30, 150) if is_potentially_risky else random.uniform(0.1, 20), 4)
    gas = round(random.uniform(50, 150) if is_potentially_risky else random.uniform(20, 100), 2)

    tx = {
        "tx_hash": f"0x{random.randint(10**63, 10**64-1):064x}",
        "from_address": from_addr,
        "to_address": to_addr,
        "value_eth": value,
        "gas_price": gas,
        "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ') # Use current time
    }
    return tx

def append_to_csv(filepath, transaction_dict, fieldnames):
    """Appends a single transaction dictionary to a CSV file."""
    file_exists = os.path.isfile(filepath)
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader() # Write header only if file is new/empty
            writer.writerow(transaction_dict)
    except IOError as e:
        print(f"Error appending to CSV {filepath}: {e}")

def update_csv_status(filepath, tx_hash, new_status, fieldnames):
    """Updates the status of a specific transaction in the CSV."""
    # This is inefficient for large files - Database is much better!
    try:
        df = pd.read_csv(filepath)
        if tx_hash not in df['tx_hash'].values:
            print(f"Warning: tx_hash {tx_hash} not found in {filepath} for status update.")
            return False # Indicate failure

        df.loc[df['tx_hash'] == tx_hash, 'final_status'] = new_status
        # Add review timestamp if needed (requires column)
        # df.loc[df['tx_hash'] == tx_hash, 'review_timestamp'] = datetime.now().isoformat()
        df.to_csv(filepath, index=False)
        # Find the updated row to return
        updated_row = df[df['tx_hash'] == tx_hash].iloc[0].to_dict()
         # Convert types back like in get_flagged_transactions
        updated_row['value_eth'] = float(updated_row.get('value_eth', 0))
        updated_row['gas_price'] = float(updated_row.get('gas_price', 0))
        updated_row['final_score'] = float(updated_row.get('final_score', 0))
        updated_row['agent_1_score'] = float(updated_row.get('agent_1_score', 0))
        updated_row['agent_2_score'] = float(updated_row.get('agent_2_score', 0))
        updated_row['agent_3_score'] = float(updated_row.get('agent_3_score', 0)) # Added agent_3
        return updated_row # Return the updated transaction data
    except pd.errors.EmptyDataError:
        print(f"Warning: {filepath} is empty. Cannot update status.")
        return False
    except FileNotFoundError:
        print(f"Error: {filepath} not found for status update.")
        return False
    except Exception as e:
        print(f"Error updating CSV status: {e}")
        return False


# --- Real-time Simulation Background Task ---
def run_realtime_simulation():
    """Background task to simulate and process transactions."""
    global simulation_running
    print("Starting real-time simulation background task...")
    simulation_running = True

    # Define CSV fields including agent_3
    csv_fieldnames = [
        'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
        'timestamp', 'final_status', 'final_score', 'reasons',
        'agent_1_score', 'agent_2_score', 'agent_3_score'
    ]
    # Ensure header exists if file doesn't
    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
         append_to_csv(OUTPUT_FILE, {}, csv_fieldnames) # Creates file with header

    while simulation_running:
        try:
            # 1. Generate or fetch a new transaction
            new_tx = generate_simulated_transaction()
            print(f"\nProcessing Tx: {new_tx['tx_hash'][:10]}... Value: {new_tx['value_eth']:.2f} ETH")

            # 2. Process through agent pipeline
            processed_tx = process_transaction_pipeline(new_tx.copy()) # Use the imported pipeline function

            # 3. Check status and emit if flagged/denied
            status = processed_tx.get('final_status')
            if status in ['FLAG_FOR_REVIEW', 'DENY']:
                print(f"  -> FLAGGED ({status}) Score: {processed_tx.get('final_score', 0):.1f}. Emitting event.")
                # Append to our persistent log (CSV in this case)
                append_to_csv(OUTPUT_FILE, processed_tx, csv_fieldnames)
                # Emit to connected clients
                socketio.emit('new_flagged_transaction', processed_tx)
            else:
                print(f"  -> Approved. Score: {processed_tx.get('final_score', 0):.1f}")

            # 4. Wait before processing the next one
            # Use socketio.sleep for cooperative yielding in eventlet/gevent
            socketio.sleep(random.uniform(2, 6)) # Wait 2-6 seconds

        except Exception as e:
            print(f"Error in simulation loop: {e}")
            socketio.sleep(5) # Wait longer after an error

    print("Real-time simulation background task stopped.")

# --- SocketIO Events ---

@socketio.on('connect')
def handle_connect():
    global simulation_task, simulation_running
    print('Client connected:', request.sid)
    # Start the simulation loop in the background if it's not already running
    # This ensures it runs only when at least one client is connected
    if not simulation_running and simulation_task is None:
        print("First client connected, starting simulation task.")
        # Make sure environment is suitable for background tasks
        if socketio.async_mode in ['eventlet', 'gevent', 'gevent_uwsgi']:
             simulation_task = socketio.start_background_task(target=run_realtime_simulation)
        else:
             print("Warning: SocketIO async_mode not set to eventlet or gevent. Background task might block.")
             # For development/testing with Flask server, threading might be okay, but less robust
             # import threading
             # simulation_task = threading.Thread(target=run_realtime_simulation)
             # simulation_task.start()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)
    # Optional: Stop simulation if no clients are connected
    # Might want to keep it running depending on requirements
    # if not socketio.server.eio.sockets: # Check if any clients remain
    #     global simulation_running, simulation_task
    #     if simulation_running:
    #         print("Last client disconnected, stopping simulation task.")
    #         simulation_running = False
    #         # Need a way to properly stop the loop if using threads
    #         # For eventlet/gevent, setting simulation_running to False is often enough
    #         simulation_task = None


# --- API Endpoints ---

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check status including simulation running state."""
    db_exists = os.path.exists(DB_FILE)
    threat_list_exists = os.path.exists(THREAT_FILE)
    # Check if the output file has more than just a header (simplistic check)
    output_has_data = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 150 # Approx header size

    return jsonify({
        'database_ready': db_exists,
        'simulation_running': simulation_running, # Use the global flag
        'threat_list_available': threat_list_exists,
        'has_flagged_data': output_has_data # Indicate if any data exists to be loaded
    })

@app.route('/api/setup', methods=['POST'])
def run_setup():
    """Run the 0_setup_database.py script."""
    # (Keep this endpoint as is - setup is still a one-time action)
    try:
        print("Running database setup...")
        result = subprocess.run(
            [PYTHON_EXE, '0_setup_database.py'],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        # Ensure the header exists for the output CSV after setup
        csv_fieldnames = [
            'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
            'timestamp', 'final_status', 'final_score', 'reasons',
            'agent_1_score', 'agent_2_score', 'agent_3_score'
        ]
        if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
             append_to_csv(OUTPUT_FILE, {}, csv_fieldnames) # Creates file with header

        return jsonify({"message": "Database and models set up successfully"}), 200
    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e.stderr}")
        return jsonify({"error": f"An error occurred during setup: {e.stderr}"}), 500
    except FileNotFoundError:
        return jsonify({"error": "Python executable not found. Make sure python is in your PATH."}), 500

# REMOVED: /api/run-simulation POST endpoint
# The simulation now starts automatically when a client connects via WebSocket

@app.route('/api/flagged-transactions', methods=['GET'])
def get_flagged_transactions():
    """Read the initial CSV results and return them as JSON."""
    # This now just reads the existing log file on initial load
    if not os.path.exists(OUTPUT_FILE):
         print(f"Warning: {OUTPUT_FILE} not found on initial load request.")
         return jsonify([]) # Return empty if no file yet

    transactions = []
    try:
        # Use pandas for easier reading, especially with potentially large files
        df = pd.read_csv(OUTPUT_FILE)
        if df.empty:
            return jsonify([])

        # Convert to dictionary records and handle potential NaN/types
        df = df.fillna(0) # Replace NaN with 0 for scores, etc.
        transactions = df.to_dict('records')

        # Convert types like before
        for row in transactions:
             row['value_eth'] = float(row.get('value_eth', 0))
             row['gas_price'] = float(row.get('gas_price', 0))
             row['final_score'] = float(row.get('final_score', 0))
             row['agent_1_score'] = float(row.get('agent_1_score', 0))
             row['agent_2_score'] = float(row.get('agent_2_score', 0))
             row['agent_3_score'] = float(row.get('agent_3_score', 0))
             row['reasons'] = str(row.get('reasons', '')) # Ensure reasons is string

        return jsonify(transactions)
    except pd.errors.EmptyDataError:
         return jsonify([]) # File exists but is empty
    except Exception as e:
        print(f"Error reading {OUTPUT_FILE}: {e}")
        return jsonify({"error": f"An error occurred reading the file: {str(e)}"}), 500

# --- Wallet Profile Endpoint (Remains the same) ---
@app.route('/api/wallet/<address>', methods=['GET'])
def get_wallet_profile(address):
    # ... (keep implementation as before) ...
    if not os.path.exists(DB_FILE):
        return jsonify({"error": f"{DB_FILE} not found. Run setup."}), 404
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM wallet_profiles WHERE wallet_address = ?", (address.lower(),))
        profile = cursor.fetchone()
        if profile:
            return jsonify(dict(profile))
        else:
            return jsonify({"message": "Wallet profile not found"}), 404
    except sqlite3.Error as e:
        print(f"Database error fetching wallet profile: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        print(f"Error fetching wallet profile: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


# --- Analyst Review Endpoint (Modified to emit WebSocket event) ---
@app.route('/api/review/<tx_hash>', methods=['POST'])
def submit_review(tx_hash):
    """Allow manual override and emit update event."""
    data = request.get_json()
    new_status = data.get('status')
    analyst_id = data.get('analyst_id', 'manual_override')

    if not new_status or new_status not in ['APPROVE', 'DENY', 'FLAG_FOR_REVIEW']:
        return jsonify({"error": "Invalid status. Must be APPROVE, DENY, or FLAG_FOR_REVIEW."}), 400

    if not os.path.exists(OUTPUT_FILE):
        return jsonify({"error": "Flagged transactions file not found."}), 404

    csv_fieldnames = [ # Ensure these match the current file structure
        'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
        'timestamp', 'final_status', 'final_score', 'reasons',
        'agent_1_score', 'agent_2_score', 'agent_3_score'
    ]

    # Update CSV (still inefficient, but consistent for now)
    updated_transaction = update_csv_status(OUTPUT_FILE, tx_hash, new_status, csv_fieldnames)

    if updated_transaction:
        # Emit update to all connected clients
        print(f"Review submitted for {tx_hash}. New status: {new_status}. Emitting update.")
        socketio.emit('transaction_update', updated_transaction)
        return jsonify({"message": f"Transaction {tx_hash} status updated to {new_status}"}), 200
    else:
        # update_csv_status handles logging errors
        return jsonify({"error": "Failed to update transaction status or transaction not found."}), 500

    # --- Database approach (preferred) ---
    # try:
    #     conn = # ... connect to your DB
    #     cursor = conn.cursor()
    #     # Update the status
    #     cursor.execute("UPDATE transactions SET final_status = ? WHERE tx_hash = ?", (new_status, tx_hash))
    #     if cursor.rowcount == 0:
    #         conn.close()
    #         return jsonify({"error": "Transaction not found"}), 404
    #     # Fetch the updated row to emit
    #     cursor.execute("SELECT * FROM transactions WHERE tx_hash = ?", (tx_hash,))
    #     updated_row = cursor.fetchone() # Fetch as dict/object
    #     conn.commit()
    #     conn.close()
    #     socketio.emit('transaction_update', updated_row) # Convert row to dict if needed
    #     return jsonify({"message": "Update successful"}), 200
    # except Exception as e:
    #     # handle error
    #     return jsonify({"error": "DB update failed"}), 500


# --- Threat List Management Endpoints (Remain the same) ---
@app.route('/api/threats', methods=['GET'])
def get_threats():
    # ... (keep implementation as before) ...
    try:
        wallets = read_threat_list()
        return jsonify(wallets)
    except Exception as e:
        print(f"Error reading threat list: {e}")
        return jsonify({"error": f"Could not read threat list: {e}"}), 500

@app.route('/api/threats', methods=['POST'])
def add_threat():
    # ... (keep implementation as before) ...
    data = request.get_json()
    new_wallet = data.get('wallet_address')
    if not new_wallet or not isinstance(new_wallet, str): return jsonify({"error": "Invalid wallet"}), 400
    new_wallet = new_wallet.strip().lower()
    if not new_wallet.startswith('0x') or len(new_wallet) != 42: return jsonify({"error": "Invalid format"}), 400
    try:
        current_wallets = read_threat_list()
        if new_wallet in current_wallets: return jsonify({"message": "Exists"}), 200
        current_wallets.append(new_wallet)
        write_threat_list(current_wallets)
        # Emit event that threat list was updated (optional)
        socketio.emit('threat_list_updated', read_threat_list())
        return jsonify({"message": f"Wallet {new_wallet} added."}), 201
    except Exception as e: return jsonify({"error": f"Update failed: {e}"}), 500

@app.route('/api/threats', methods=['DELETE'])
def remove_threat():
    # ... (keep implementation as before) ...
    data = request.get_json()
    wallet_to_remove = data.get('wallet_address')
    if not wallet_to_remove or not isinstance(wallet_to_remove, str): return jsonify({"error": "Invalid wallet"}), 400
    wallet_to_remove = wallet_to_remove.strip().lower()
    try:
        current_wallets = read_threat_list()
        if wallet_to_remove not in current_wallets: return jsonify({"error": "Not found"}), 404
        updated_wallets = [w for w in current_wallets if w != wallet_to_remove]
        write_threat_list(updated_wallets)
        # Emit event that threat list was updated (optional)
        socketio.emit('threat_list_updated', updated_wallets)
        return jsonify({"message": f"Wallet {wallet_to_remove} removed."}), 200
    except Exception as e: return jsonify({"error": f"Update failed: {e}"}), 500


# --- Main Execution ---
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    if not os.path.exists(DB_FILE) or not os.path.exists(THREAT_FILE):
        print("-" * 60)
        print("Warning: Essential data files missing. Run setup.")
        print("-" * 60)

    print("Starting Flask-SocketIO server with eventlet...")
    # Use socketio.run() to start the server correctly with WebSocket support
    socketio.run(app, debug=True, port=5000, host='0.0.0.0') # Use 0.0.0.0 to be accessible on network