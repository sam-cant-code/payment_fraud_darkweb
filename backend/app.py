import csv
import json
import os
import subprocess  # Import the subprocess module
from flask import Flask, jsonify, abort
from flask_cors import CORS

# We no longer need to import the other scripts, so the warnings are gone.
app = Flask(__name__)
# Enable CORS to allow requests from your React app (localhost:3000)
CORS(app)

OUTPUT_FILE = 'output/flagged_transactions.csv'
DB_FILE = 'data/wallet_profiles.db'
PYTHON_EXE = 'python' # or 'python3' if 'python' doesn't work

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check the status of required data and model files."""
    db_exists = os.path.exists(DB_FILE)
    output_exists = os.path.exists(OUTPUT_FILE)
    return jsonify({
        'database_ready': db_exists,
        'simulation_run': output_exists
    })

@app.route('/api/setup', methods=['POST'])
def run_setup():
    """Run the 0_setup_database.py script."""
    try:
        print("Running database setup...")
        # Use subprocess.run to execute the script
        result = subprocess.run(
            [PYTHON_EXE, '0_setup_database.py'], 
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        return jsonify({"message": "Database and models set up successfully"}), 200
    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e.stderr}")
        return jsonify({"error": f"An error occurred during setup: {e.stderr}"}), 500
    except FileNotFoundError:
        return jsonify({"error": "Python executable not found. Make sure python is in your PATH."}), 500

@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    """Run the 1_run_simulation.py script."""
    try:
        print("Running simulation...")
        # Use subprocess.run to execute the script
        result = subprocess.run(
            [PYTHON_EXE, '1_run_simulation.py'], 
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        return jsonify({"message": "Simulation completed successfully"}), 200
    except subprocess.CalledProcessError as e:
        print(f"Error during simulation: {e.stderr}")
        return jsonify({"error": f"An error occurred during simulation: {e.stderr}"}), 500
    except FileNotFoundError:
        return jsonify({"error": "Python executable not found. Make sure python is in your PATH."}), 500

@app.route('/api/flagged-transactions', methods=['GET'])
def get_flagged_transactions():
    """Read the CSV results and return them as JSON."""
    if not os.path.exists(OUTPUT_FILE):
        return jsonify({"error": "File not found. Run simulation first."}), 404
    
    transactions = []
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numerical strings to numbers for better JSON handling
                row['value_eth'] = float(row.get('value_eth', 0))
                row['gas_price'] = float(row.get('gas_price', 0))
                row['final_score'] = float(row.get('final_score', 0))
                row['agent_1_score'] = float(row.get('agent_1_score', 0))
                row['agent_2_score'] = float(row.get('agent_2_score', 0))
                transactions.append(row)
        
        return jsonify(transactions)
    except Exception as e:
        return jsonify({"error": f"An error occurred reading the file: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure data/models exist before starting
    if not os.path.exists(DB_FILE):
        print(f"Warning: {DB_FILE} not found. Run 'setup_backend.bat' first.")
    
    app.run(debug=True, port=5000)