"""
Simulation Runner for Blockchain Fraud Detection MVP
Processes transactions through the agent pipeline (for batch processing)
UPDATED: Processes only the last 1000 transactions for the dashboard.
"""

import json
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# --- PATH CONFIGURATION ---
# backend/scripts/run_simulation.py
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # project root

# Add parent directory to path so we can import from app
sys.path.insert(0, str(BACKEND_DIR))

try:
    from app.agents import process_transaction_pipeline
except ImportError:
    print("\n[FATAL ERROR] Could not import 'app.agents'.")
    print("Please ensure you are running this script from the project root folder.")
    print("Example: python -m backend.scripts.run_simulation")
    sys.exit(1)

# *** MODIFICATION: Define how many recent transactions to simulate for dashboard ***
NUM_TRANSACTIONS_FOR_SIMULATION = 1000

def load_transactions(filepath):
    """Load transactions from JSON file"""
    try:
        with open(filepath, 'r') as f:
            transactions = json.load(f)
        print(f"Loaded {len(transactions)} transactions from {filepath}")
        return transactions
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run scripts/initialize_and_train.py first.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        return []
    except Exception as e:
        print(f"Error loading transactions: {e}")
        return []


def save_processed_transactions(transactions, filepath):
    """Save PROCESSED transactions to CSV""" # Changed comment
    if not transactions:
        print("No transactions were processed to save.")
        return

    fieldnames = [
        'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
        'timestamp', 'final_status', 'final_score', 'reasons',
        'agent_1_score', 'agent_2_score', 'agent_3_score', 'ml_fraud_probability', 'ml_prediction', 'is_fraud'
    ]
    
    os.makedirs(filepath.parent, exist_ok=True)
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            rows_to_write = []
            for tx in transactions:
                # Ensure all keys exist to avoid errors, defaulting to ''
                row = {key: tx.get(key, '') for key in fieldnames}
                rows_to_write.append(row)
            writer.writerows(rows_to_write)
        # *** MODIFICATION: Updated print statement ***
        print(f"\nSaved {len(transactions)} SIMULATED transactions to {filepath}")
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")
    except Exception as e:
        print(f"Unexpected error saving transactions: {e}")

# (print_summary and print_sample_flagged remain unchanged)
def print_summary(transactions):
    """Print summary statistics with ground truth comparison"""
    if not transactions:
        print("\nNo transactions processed to summarize.")
        return
    
    total = len(transactions)
    approved = sum(1 for tx in transactions if tx.get('final_status') == 'APPROVE')
    flagged = sum(1 for tx in transactions if tx.get('final_status') == 'FLAG_FOR_REVIEW')
    denied = sum(1 for tx in transactions if tx.get('final_status') == 'DENY')
    
    # Count ground truth if available
    has_labels = sum(1 for tx in transactions if tx.get('is_fraud') is not None)
    actual_fraud = sum(1 for tx in transactions if tx.get('is_fraud') == 1)
    
    print("\n" + "="*60)
    print("TRANSACTION PROCESSING SUMMARY")
    print("="*60)
    print(f"Total Transactions Processed in Simulation: {total}") # Updated label
    if total > 0:
        print(f"\nModel Predictions (on simulated subset):") # Updated label
        print(f"  Approved:           {approved} ({approved/total*100:.1f}%)")
        print(f"  Flagged for Review: {flagged} ({flagged/total*100:.1f}%)")
        print(f"  Denied:             {denied} ({denied/total*100:.1f}%)")
        
        if has_labels > 0:
            print(f"\nGround Truth (in simulated subset):") # Updated label
            print(f"  Labeled samples:    {has_labels}")
            print(f"  Actual fraud:       {actual_fraud} ({actual_fraud/has_labels*100:.1f}%)")
            print(f"  Actual normal:      {has_labels - actual_fraud} ({(has_labels-actual_fraud)/has_labels*100:.1f}%)")
    
    print("="*60)

def print_sample_flagged(transactions, num_samples=5):
    """Print sample flagged/denied transactions"""
    flagged_or_denied = [tx for tx in transactions if tx.get('final_status') in ['FLAG_FOR_REVIEW', 'DENY']]
    
    if not flagged_or_denied:
        print("\n✓ No transactions were flagged or denied in this simulation batch.")
        return

    print(f"\nSample Flagged/Denied Transactions (showing up to {num_samples}):")
    print("-"*60)
    
    for i, tx in enumerate(flagged_or_denied[:num_samples], 1):
        print(f"\n{i}. Tx Hash: {tx.get('tx_hash', 'N/A')[:16]}...")
        print(f"   Value: {tx.get('value_eth', 0):.4f} ETH")
        print(f"   Status: {tx.get('final_status', 'N/A')}")
        print(f"   Score: {tx.get('final_score', 0):.1f}")
        print(f"   Agent Scores (A1/A2/A3): {tx.get('agent_1_score', 0):.0f} / {tx.get('agent_2_score', 0):.0f} / {tx.get('agent_3_score', 0):.0f}")
        print(f"   Reasons: {tx.get('reasons', 'N/A')}")
        
        if 'is_fraud' in tx and tx['is_fraud'] is not None:
            actual = "FRAUD" if tx['is_fraud'] == 1 else "NORMAL"
            predicted_fraud = tx.get('final_status') in ['FLAG_FOR_REVIEW', 'DENY']
            match = "✓" if (tx['is_fraud'] == 1) == predicted_fraud else "✗"
            print(f"   Ground Truth: {actual} {match}")
        
        print("-" * 20)


def main():
    """Main simulation function"""
    start_time_sim = datetime.now()
    print("=" * 60)
    print("BATCH SIMULATION RUNNER")
    print("="*60)
    
    # Define correct input/output paths from PROJECT_ROOT
    input_filepath = PROJECT_ROOT / 'data' / 'mock_blockchain_transactions.json'
    
    model_timestamp = None
    if len(sys.argv) > 1:
        model_timestamp = sys.argv[1]
        print(f"--- Using model timestamp: {model_timestamp} ---")
        output_filepath = PROJECT_ROOT / 'output' / f'all_processed_transactions_{model_timestamp}.csv'
    else:
        print("--- Using default model (as defined in agents.py) ---")
        output_filepath = PROJECT_ROOT / 'output' / 'all_processed_transactions.csv'
    
    
    print("\nStep 1: Loading ALL Transactions...")
    all_transactions = load_transactions(input_filepath) # Load all
    
    if not all_transactions:
        print("\nError: No transactions loaded. Exiting.")
        return

    # *** MODIFICATION: Select only the last N transactions for processing ***
    transactions_to_process = all_transactions[-NUM_TRANSACTIONS_FOR_SIMULATION:]
    print(f"✓ Selected the most recent {len(transactions_to_process)} transactions for simulation.")

    print(f"\nStep 2: Processing {len(transactions_to_process)} Transactions Through Agent Pipeline...")
    processed_transactions = []
    
    # Progress bar
    total_to_process = len(transactions_to_process)
    for i, tx in enumerate(transactions_to_process, 1):
        # *** MODIFICATION: Adjust progress print for smaller number ***
        if total_to_process < 2000 or i % 100 == 0 or i == total_to_process:
            print(f"\r  Processing {i}/{total_to_process}... ({i/total_to_process*100:.1f}%)", end="")
        
        processed_tx = process_transaction_pipeline(tx.copy(), model_timestamp=model_timestamp)
        processed_transactions.append(processed_tx)

    print(f"\n✓ Processed all {len(processed_transactions)} selected transactions.")

    print("\nStep 3: Saving SIMULATED Transactions...") # Updated label
    save_processed_transactions(processed_transactions, filepath=output_filepath)

    print_summary(processed_transactions)
    print_sample_flagged(processed_transactions)

    end_time_sim = datetime.now()
    duration_sim = end_time_sim - start_time_sim
    
    print("\n" + "=" * 60)
    print(f"✓ BATCH SIMULATION COMPLETE! (Duration: {duration_sim})")
    print("=" * 60)
    print(f"Output saved to: {output_filepath}")
    
    if model_timestamp:
        print(f"\nNext step: Dashboard is launching for model {model_timestamp}")
    else:
        print(f"\nNext step: Dashboard is launching for default model")


if __name__ == "__main__":
    main()