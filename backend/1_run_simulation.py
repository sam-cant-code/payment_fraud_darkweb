"""
Simulation Runner for Blockchain Fraud Detection MVP
Processes transactions through the agent pipeline
"""

import json
import csv
from agents import agent_1_monitor, agent_2_behavioral, agent_4_decision


def load_transactions(filepath='data/mock_blockchain_transactions.json'):
    """Load transactions from JSON file"""
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


def process_transaction_pipeline(transaction):
    """
    Pass transaction through the complete agent pipeline:
    Agent 1 (Monitor) -> Agent 2 (Behavioral) -> Agent 4 (Decision)
    """
    # Agent 1: Threat Intelligence
    transaction = agent_1_monitor(transaction)
    
    # Agent 2: Behavioral Analysis
    transaction = agent_2_behavioral(transaction)
    
    # Agent 4: Final Decision
    transaction = agent_4_decision(transaction)
    
    return transaction


def save_flagged_transactions(transactions, filepath='output/flagged_transactions.csv'):
    """Save flagged and denied transactions to CSV"""
    # Filter for flagged or denied transactions
    flagged = [tx for tx in transactions if tx.get('final_status') in ['FLAG_FOR_REVIEW', 'DENY']]
    
    if not flagged:
        print("No transactions were flagged or denied.")
        return
    
    # Define CSV columns
    fieldnames = [
        'tx_hash',
        'from_address',
        'to_address',
        'value_eth',
        'gas_price',
        'timestamp',
        'final_status',
        'final_score',
        'reasons',
        'agent_1_score',
        'agent_2_score'
    ]
    
    # Write to CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(flagged)
    
    print(f"\nSaved {len(flagged)} flagged transactions to {filepath}")


def print_summary(transactions):
    """Print summary statistics"""
    total = len(transactions)
    approved = sum(1 for tx in transactions if tx.get('final_status') == 'APPROVE')
    flagged = sum(1 for tx in transactions if tx.get('final_status') == 'FLAG_FOR_REVIEW')
    denied = sum(1 for tx in transactions if tx.get('final_status') == 'DENY')
    
    print("\n" + "=" * 60)
    print("TRANSACTION PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total Transactions:     {total}")
    print(f"Approved:               {approved} ({approved/total*100:.1f}%)")
    print(f"Flagged for Review:     {flagged} ({flagged/total*100:.1f}%)")
    print(f"Denied:                 {denied} ({denied/total*100:.1f}%)")
    print("=" * 60)


def print_sample_flagged(transactions, num_samples=3):
    """Print sample flagged transactions"""
    flagged = [tx for tx in transactions if tx.get('final_status') in ['FLAG_FOR_REVIEW', 'DENY']]
    
    if not flagged:
        return
    
    print(f"\nSample Flagged Transactions (showing up to {num_samples}):")
    print("-" * 60)
    
    for i, tx in enumerate(flagged[:num_samples], 1):
        print(f"\n{i}. Transaction Hash: {tx.get('tx_hash', 'N/A')}")
        print(f"   From: {tx.get('from_address', 'N/A')}")
        print(f"   To: {tx.get('to_address', 'N/A')}")
        print(f"   Value: {tx.get('value_eth', 0)} ETH")
        print(f"   Status: {tx.get('final_status', 'N/A')}")
        print(f"   Final Score: {tx.get('final_score', 0):.2f}")
        print(f"   Reasons: {tx.get('reasons', 'N/A')}")


def main():
    """Main simulation function"""
    print("=" * 60)
    print("BLOCKCHAIN FRAUD DETECTION MVP - SIMULATION")
    print("=" * 60)
    
    # Load transactions
    print("\nStep 1: Loading Transactions...")
    transactions = load_transactions()
    
    if not transactions:
        print("\nError: No transactions to process. Exiting.")
        return
    
    # Process each transaction through the pipeline
    print("\nStep 2: Processing Transactions Through Agent Pipeline...")
    processed_transactions = []
    
    for i, tx in enumerate(transactions, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(transactions)} transactions...")
        
        processed_tx = process_transaction_pipeline(tx.copy())
        processed_transactions.append(processed_tx)
    
    print(f"  Processed all {len(processed_transactions)} transactions.")
    
    # Save flagged transactions
    print("\nStep 3: Saving Flagged Transactions...")
    save_flagged_transactions(processed_transactions)
    
    # Print summary
    print_summary(processed_transactions)
    
    # Print sample flagged transactions
    print_sample_flagged(processed_transactions)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)
    print("\nNext step:")
    print("Run the dashboard: streamlit run 2_dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()