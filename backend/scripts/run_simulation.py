"""
Simulation Runner for Blockchain Fraud Detection MVP
Processes transactions through the agent pipeline (for batch processing)
"""

import json
import csv
import os
import sys
from datetime import datetime

# Add parent directory to path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents import process_transaction_pipeline

# *** FIX: Path changed from ../data to ../../data ***
def load_transactions(filepath='../../data/mock_blockchain_transactions.json'):
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

# *** FIX: Default path changed from ../output to ../../output ***
def save_processed_transactions(transactions, filepath='../../output/all_processed_transactions.csv'):
    """Save ALL processed transactions to CSV"""
    if not transactions:
        print("No transactions were processed to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
            except IOError as e:
                print(f"Error creating empty output file: {e}")
        return

    fieldnames = [
        'tx_hash', 'from_address', 'to_address', 'value_eth', 'gas_price',
        'timestamp', 'final_status', 'final_score', 'reasons',
        'agent_1_score', 'agent_2_score', 'agent_3_score', 'is_fraud'
    ]
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            rows_to_write = []
            for tx in transactions:
                row = {**{key: '' for key in fieldnames}, **tx}
                rows_to_write.append(row)
            writer.writerows(rows_to_write)
        print(f"\nSaved {len(transactions)} processed transactions to {filepath}")
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")
    except Exception as e:
        print(f"Unexpected error saving transactions: {e}")


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
    print(f"Total Transactions Processed: {total}")
    if total > 0:
        print(f"\nModel Predictions:")
        print(f"  Approved:           {approved} ({approved/total*100:.1f}%)")
        print(f"  Flagged for Review: {flagged} ({flagged/total*100:.1f}%)")
        print(f"  Denied:             {denied} ({denied/total*100:.1f}%)")
        
        if has_labels > 0:
            print(f"\nGround Truth (from mock data):")
            print(f"  Labeled samples:    {has_labels}")
            print(f"  Actual fraud:       {actual_fraud} ({actual_fraud/has_labels*100:.1f}%)")
            print(f"  Actual normal:      {has_labels - actual_fraud} ({(has_labels-actual_fraud)/has_labels*100:.1f}%)")
            print(f"\nNote: Run 'python scripts/calculate_accuracy.py <TIMESTAMP>' for detailed metrics")
    
    print("="*60)


def print_sample_flagged(transactions, num_samples=5):
    """Print sample flagged/denied transactions"""
    flagged_or_denied = [tx for tx in transactions if tx.get('final_status') in ['FLAG_FOR_REVIEW', 'DENY']]
    
    if not flagged_or_denied:
        print("\n✓ No transactions were flagged or denied in this batch.")
        print("  All transactions passed security checks!")
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
    
    # *** MODIFICATION: Check for timestamp argument ***
    model_timestamp = None
    if len(sys.argv) > 1:
        model_timestamp = sys.argv[1]
        print(f"--- Using model timestamp: {model_timestamp} ---")
        # *** FIX: Path changed from ../output to ../../output ***
        output_filepath = f'../../output/all_processed_transactions_{model_timestamp}.csv'
    else:
        print("--- Using default model (as defined in agents.py) ---")
        # *** FIX: Path changed from ../output to ../../output ***
        output_filepath = '../../output/all_processed_transactions.csv'
    
    
    print("\nStep 1: Loading Transactions...")
    transactions = load_transactions()
    
    if not transactions:
        print("\nError: No transactions loaded. Exiting.")
        save_processed_transactions([], filepath=output_filepath) # Pass filepath
        return

    print(f"\nStep 2: Processing {len(transactions)} Transactions Through Agent Pipeline...")
    processed_transactions = []
    
    # Progress bar
    total = len(transactions)
    for i, tx in enumerate(transactions, 1):
        if i % 100 == 0 or i == total:
            print(f"\r  Processing {i}/{total}... ({i/total*100:.1f}%)", end="")
        
        # *** MODIFICATION: Pass timestamp to pipeline ***
        processed_tx = process_transaction_pipeline(tx.copy(), model_timestamp=model_timestamp)
        processed_transactions.append(processed_tx)

    print(f"\n✓ Processed all {len(processed_transactions)} transactions.")

    print("\nStep 3: Saving ALL Processed Transactions...")
    # *** MODIFICATION: Pass dynamic filepath ***
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
        print(f"\nNext step: Run 'python scripts/calculate_accuracy.py {model_timestamp}' to see model performance")
    else:
        print(f"\nNext step: Run 'python scripts/calculate_accuracy.py' to see model performance")


if __name__ == "__main__":
    main()