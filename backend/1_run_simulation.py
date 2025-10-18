"""
Simulation Runner for Blockchain Fraud Detection MVP
Processes transactions through the agent pipeline (for batch processing)
"""

import json
import csv
import os
# Import the pipeline function FROM agents.py
from agents import process_transaction_pipeline # <-- Corrected import

def load_transactions(filepath='data/mock_blockchain_transactions.json'):
    """Load transactions from JSON file"""
    # ... (function remains the same as before) ...
    try:
        with open(filepath, 'r') as f:
            transactions = json.load(f)
        print(f"Loaded {len(transactions)} transactions from {filepath}")
        return transactions
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run 0_setup_database.py first.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        return []

# REMOVED: process_transaction_pipeline function definition (it's now imported)

def save_flagged_transactions(transactions, filepath='output/flagged_transactions.csv'):
    """Save flagged and denied transactions to CSV"""
    # ... (function remains the same as before, ensures agent_3_score column) ...
    flagged = [tx for tx in transactions if tx.get('final_status') in ['FLAG_FOR_REVIEW', 'DENY']]
    if not flagged:
        print("No transactions were flagged or denied.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
             default_headers = [ 'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price', 'timestamp', 'final_status', 'final_score', 'reasons', 'agent_1_score', 'agent_2_score', 'agent_3_score' ]
             try:
                 with open(filepath, 'w', newline='') as f: writer = csv.writer(f); writer.writerow(default_headers)
                 print(f"Created empty output file with headers: {filepath}")
             except IOError as e: print(f"Error creating empty output file: {e}")
        return
    fieldnames = [ 'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price', 'timestamp', 'final_status', 'final_score', 'reasons', 'agent_1_score', 'agent_2_score', 'agent_3_score' ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(flagged)
        print(f"\nSaved {len(flagged)} flagged transactions to {filepath}")
    except IOError as e: print(f"Error writing to {filepath}: {e}")


def print_summary(transactions):
    """Print summary statistics"""
    # ... (function remains the same as before) ...
    total = len(transactions); approved = sum(1 for tx in transactions if tx.get('final_status') == 'APPROVE'); flagged = sum(1 for tx in transactions if tx.get('final_status') == 'FLAG_FOR_REVIEW'); denied = sum(1 for tx in transactions if tx.get('final_status') == 'DENY')
    print("\n" + "="*60 + "\nTRANSACTION PROCESSING SUMMARY\n" + "="*60); print(f"Total Transactions:     {total}"); print(f"Approved:               {approved} ({approved/total*100:.1f}%)" if total else ""); print(f"Flagged for Review:     {flagged} ({flagged/total*100:.1f}%)" if total else ""); print(f"Denied:                 {denied} ({denied/total*100:.1f}%)" if total else ""); print("="*60)


def print_sample_flagged(transactions, num_samples=3):
    """Print sample flagged transactions"""
    # ... (function remains the same as before) ...
    flagged = [tx for tx in transactions if tx.get('final_status') in ['FLAG_FOR_REVIEW', 'DENY']];
    if not flagged: return
    print(f"\nSample Flagged Transactions (showing up to {num_samples}):\n" + "-"*60)
    for i, tx in enumerate(flagged[:num_samples], 1): print(f"\n{i}. Tx Hash: {tx.get('tx_hash', 'N/A')[:16]}..."); print(f"   Value: {tx.get('value_eth', 0):.4f} ETH"); print(f"   Status: {tx.get('final_status', 'N/A')}"); print(f"   Score: {tx.get('final_score', 0):.1f}"); print(f"   Scores (A1/A2/A3): {tx.get('agent_1_score', 0):.0f} / {tx.get('agent_2_score', 0):.0f} / {tx.get('agent_3_score', 0):.0f}"); print(f"   Reasons: {tx.get('reasons', 'N/A')}")

def main():
    """Main simulation function"""
    print("=" * 60 + "\nBATCH SIMULATION RUNNER\n" + "=" * 60)
    print("\nStep 1: Loading Transactions...")
    transactions = load_transactions()
    if not transactions:
        print("\nError: No transactions to process. Exiting.")
        save_flagged_transactions([])
        return
    print("\nStep 2: Processing Transactions Through Agent Pipeline...")
    processed_transactions = []
    for i, tx in enumerate(transactions, 1):
        print(f"\r  Processing {i}/{len(transactions)}...", end="")
        # Now calls the imported function
        processed_tx = process_transaction_pipeline(tx.copy())
        processed_transactions.append(processed_tx)
    print(f"\n  Processed all {len(processed_transactions)} transactions.")
    print("\nStep 3: Saving Flagged Transactions...")
    save_flagged_transactions(processed_transactions)
    print_summary(processed_transactions)
    print_sample_flagged(processed_transactions)
    print("\n" + "=" * 60 + "\nBATCH SIMULATION COMPLETE!\n" + "=" * 60)

if __name__ == "__main__":
    main()