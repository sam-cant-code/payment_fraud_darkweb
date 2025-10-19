"""
Accuracy Calculation Script for Blockchain Fraud Detection MVP
Reads the processed transactions CSV and compares predictions to ground truth labels.
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import sys
import numpy as np

# Add parent directory to path (in case we need any utilities)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
INPUT_FILE = '../output/all_processed_transactions.csv'

# Define how system status maps to a binary prediction
STATUS_TO_PREDICTION = {
    'DENY': 1,
    'FLAG_FOR_REVIEW': 1,
    'APPROVE': 0
}

def calculate_accuracy():
    print("=" * 60)
    print("FRAUD DETECTION ACCURACY CALCULATION")
    print("=" * 60)

    # Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ Error: Input file not found at '{INPUT_FILE}'")
        print("Please ensure the simulation has run successfully.")
        print("Run: python scripts/run_simulation.py")
        return

    try:
        df = pd.read_csv(INPUT_FILE, dtype={'is_fraud': 'Int64'})
        print(f"\n✓ Loaded {len(df)} records from {INPUT_FILE}")
    except pd.errors.EmptyDataError:
        print(f"\n❌ Error: The input file '{INPUT_FILE}' is empty.")
        return
    except Exception as e:
        print(f"\n❌ Error loading CSV file: {e}")
        return

    # Filter records with ground truth labels
    df_labeled = df.dropna(subset=['is_fraud'])
    print(f"✓ Found {len(df_labeled)} records with ground truth labels ('is_fraud').")

    if df_labeled.empty:
        print("\n❌ No records with ground truth labels found. Cannot calculate accuracy.")
        print("Make sure you ran the setup script to generate labeled mock data.")
        return

    # Extract ground truth and predictions
    y_true = df_labeled['is_fraud'].astype(int)
    y_pred = df_labeled['final_status'].map(STATUS_TO_PREDICTION).fillna(0).astype(int)

    # Get unique labels
    labels = sorted(pd.unique(y_true.tolist() + y_pred.tolist()))
    target_names = ['Normal (Class 0)', 'Fraud (Class 1)']

    if len(labels) == 1:
        if labels[0] == 0:
            target_names = ['Normal (Class 0)']
        else:
            target_names = ['Fraud (Class 1)']

    # Calculate Metrics
    print("\n" + "=" * 60)
    print("ACCURACY METRICS")
    print("=" * 60)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✓ Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Additional metrics
    if len(labels) == 2:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"✓ Precision (Fraud): {precision:.4f} ({precision * 100:.2f}%)")
        print(f"✓ Recall (Fraud):    {recall:.4f} ({recall * 100:.2f}%)")
        print(f"✓ F1-Score (Fraud):  {f1:.4f} ({f1 * 100:.2f}%)")

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)
    
    if len(labels) == 2:
        print("\nInterpretation:")
        print("  [[TN  FP]     TN = True Negatives  (Correctly approved)")
        print("   [FN  TP]]    FP = False Positives (Incorrectly flagged)")
        print("                FN = False Negatives (Missed fraud)")
        print("                TP = True Positives  (Correctly caught fraud)")
        print()
        print(f"  True Negatives  (TN): {cm[0, 0]:4d} - Correctly identified as normal")
        print(f"  False Positives (FP): {cm[0, 1]:4d} - Normal flagged as fraud")
        print(f"  False Negatives (FN): {cm[1, 0]:4d} - Fraud missed by system ⚠️")
        print(f"  True Positives  (TP): {cm[1, 1]:4d} - Fraud correctly detected ✓")
        
        # Calculate rates
        if cm[0, 0] + cm[0, 1] > 0:
            fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
            print(f"\n  False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
        if cm[1, 0] + cm[1, 1] > 0:
            fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
            print(f"  False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")

    print("\n" + "-" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 60)
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print(report)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Model tested on {len(df_labeled)} transactions")
    print(f"✓ Accuracy: {accuracy*100:.2f}%")
    
    if len(labels) == 2:
        print(f"✓ Successfully detected {cm[1, 1]}/{cm[1, 0] + cm[1, 1]} fraud cases")
        print(f"✓ False alarms: {cm[0, 1]} out of {cm[0, 0] + cm[0, 1]} normal transactions")
        
        if cm[1, 0] > 0:
            print(f"\n⚠️  WARNING: {cm[1, 0]} fraud cases were MISSED by the system!")
        else:
            print(f"\n✓ Perfect fraud detection! No fraud cases were missed.")
    
    print("\nNote: These metrics are based on simulated ground truth in mock data.")
    print("      'Fraud' predictions include both DENY and FLAG_FOR_REVIEW statuses.")
    print("=" * 60)


if __name__ == "__main__":
    calculate_accuracy()