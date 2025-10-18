"""
Simulation Runner for Blockchain Fraud Detection MVP
Processes transactions through the agent pipeline (for batch processing)
"""

import json
import csv
import os
from datetime import datetime # Added for potential use, though not strictly needed now
# Import the pipeline function FROM agents.py
from agents import process_transaction_pipeline

def load_transactions(filepath='data/mock_blockchain_transactions.json'):
    """Load transactions from JSON file (expects 'is_fraud' label now)"""
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
    except Exception as e:
        print(f"Error loading transactions: {e}")
        return []

# CORRECTED function to save ALL processed transactions
def save_processed_transactions(transactions, filepath='output/all_processed_transactions.csv'):
    """Save ALL processed transactions to CSV"""
    # Use the transactions list directly, DO NOT filter here
    processed_txs = transactions # Use the list as is

    if not processed_txs:
        print("No transactions were processed to save.")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Create an empty file with headers if it doesn't exist
        if not os.path.exists(filepath):
             default_headers = [
                 'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
                 'timestamp', 'final_status', 'final_score', 'reasons',
                 'agent_1_score', 'agent_2_score', 'agent_3_score', 'is_fraud'
             ]
             try:
                 with open(filepath, 'w', newline='', encoding='utf-8') as f:
                     writer = csv.writer(f)
                     writer.writerow(default_headers)
                 print(f"Created empty output file with headers: {filepath}")
             except IOError as e: print(f"Error creating empty output file: {e}")
        return

    # Define fieldnames including is_fraud
    fieldnames = [
        'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
        'timestamp', 'final_status', 'final_score', 'reasons',
        'agent_1_score', 'agent_2_score', 'agent_3_score', 'is_fraud'
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Use extrasaction='ignore' to handle potential extra fields gracefully
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            # Ensure 'is_fraud' key exists in each dict, default to empty string if missing
            rows_to_write = []
            for tx in processed_txs:
                row = {**{key: '' for key in fieldnames}, **tx} # Default keys to empty string
                rows_to_write.append(row)
            writer.writerows(rows_to_write)
        print(f"\nSaved {len(processed_txs)} processed transactions to {filepath}")
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")
    except Exception as e:
         print(f"Unexpected error saving transactions: {e}")


def print_summary(transactions):
    """Print summary statistics"""
    if not transactions:
        print("\nNo transactions processed to summarize.")
        return
    total = len(transactions)
    approved = sum(1 for tx in transactions if tx.get('final_status') == 'APPROVE')
    flagged = sum(1 for tx in transactions if tx.get('final_status') == 'FLAG_FOR_REVIEW')
    denied = sum(1 for tx in transactions if tx.get('final_status') == 'DENY')
    print("\n" + "="*60 + "\nTRANSACTION PROCESSING SUMMARY\n" + "="*60)
    print(f"Total Transactions Processed: {total}")
    if total > 0:
        print(f"Approved:               {approved} ({approved/total*100:.1f}%)")
        print(f"Flagged for Review:     {flagged} ({flagged/total*100:.1f}%)")
        print(f"Denied:                 {denied} ({denied/total*100:.1f}%)")
    print("="*60)


def print_sample_flagged(transactions, num_samples=5): # Increased samples
    """Print sample flagged/denied transactions"""
    # Filter only for flagged/denied here, just for printing samples
    flagged_or_denied = [tx for tx in transactions if tx.get('final_status') in ['FLAG_FOR_REVIEW', 'DENY']]
    if not flagged_or_denied:
        print("\nNo transactions were flagged or denied in this batch.")
        return

    print(f"\nSample Flagged/Denied Transactions (showing up to {num_samples}):\n" + "-"*60)
    for i, tx in enumerate(flagged_or_denied[:num_samples], 1):
        print(f"\n{i}. Tx Hash: {tx.get('tx_hash', 'N/A')[:16]}...")
        print(f"   Value: {tx.get('value_eth', 0):.4f} ETH")
        print(f"   Status: {tx.get('final_status', 'N/A')}")
        print(f"   Score: {tx.get('final_score', 0):.1f}")
        print(f"   Scores (A1/A2/A3): {tx.get('agent_1_score', 0):.0f} / {tx.get('agent_2_score', 0):.0f} / {tx.get('agent_3_score', 0):.0f}")
        print(f"   Reasons: {tx.get('reasons', 'N/A')}")
        # Print is_fraud label if present (from mock data)
        if 'is_fraud' in tx and tx['is_fraud'] is not None:
             print(f"   Ground Truth (is_fraud): {tx['is_fraud']}")
        print("-" * 20) # Separator

def main():
    """Main simulation function"""
    start_time_sim = datetime.now()
    print("=" * 60 + "\nBATCH SIMULATION RUNNER\n" + "=" * 60)
    print("\nStep 1: Loading Transactions...")
    # Ensure it loads the JSON with is_fraud labels
    transactions = load_transactions('data/mock_blockchain_transactions.json')
    if not transactions:
        print("\nError: No transactions loaded. Exiting.")
        save_processed_transactions([]) # Ensure output file exists even if empty
        return

    print(f"\nStep 2: Processing {len(transactions)} Transactions Through Agent Pipeline...")
    processed_transactions = []
    for i, tx in enumerate(transactions, 1):
        print(f"\r  Processing {i}/{len(transactions)}...", end="")
        # Ensure the pipeline preserves 'is_fraud'
        # Pass a copy to avoid modifying the original list elements if needed elsewhere
        processed_tx = process_transaction_pipeline(tx.copy())
        processed_transactions.append(processed_tx)

    print(f"\n  Processed all {len(processed_transactions)} transactions.")

    print("\nStep 3: Saving ALL Processed Transactions...")
    # Call the corrected save function (using the default output filename)
    save_processed_transactions(processed_transactions)

    # Print summary and samples
    print_summary(processed_transactions)
    print_sample_flagged(processed_transactions)

    end_time_sim = datetime.now()
    duration_sim = end_time_sim - start_time_sim
    print("\n" + "=" * 60 + f"\nBATCH SIMULATION COMPLETE! (Duration: {duration_sim})\n" + "=" * 60)
    print("Output saved to output/all_processed_transactions.csv")
    print("You can now run 'python 3_calculate_accuracy.py' to evaluate performance on this data.")

if __name__ == "__main__":
    main()