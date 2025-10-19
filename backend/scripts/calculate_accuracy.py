"""
Accuracy Calculation Script for Blockchain Fraud Detection MVP
Reads the processed transactions CSV and compares predictions to ground truth labels.
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import numpy as np

# --- Configuration ---
# File containing processed transactions with predictions and ground truth
INPUT_FILE = '../output/all_processed_transactions.csv' # <-- FIXED PATH

# Define how system status maps to a binary prediction (1=Fraud/Risky, 0=Not Fraud)
# We consider both DENY and FLAG_FOR_REVIEW as positive predictions for this calculation
STATUS_TO_PREDICTION = {
    'DENY': 1,
    'FLAG_FOR_REVIEW': 1,
    'APPROVE': 0
}

# --- Main Calculation Function ---
def calculate_accuracy():
    print("=" * 60)
    print("Fraud Detection Accuracy Calculation")
    print("=" * 60)

    # --- Load Data ---
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at '{INPUT_FILE}'")
        print("Please ensure the backend has run and generated the output file.")
        return

    try:
        # Load the CSV, explicitly handle 'is_fraud' potentially being empty/NA
        df = pd.read_csv(INPUT_FILE, dtype={'is_fraud': 'Int64'}) # Use nullable integer type
        print(f"Loaded {len(df)} records from {INPUT_FILE}")
    except pd.errors.EmptyDataError:
        print(f"Error: The input file '{INPUT_FILE}' is empty.")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # --- Data Preparation ---
    # Filter out records where ground truth is missing (e.g., live data)
    df_labeled = df.dropna(subset=['is_fraud'])
    print(f"Found {len(df_labeled)} records with ground truth labels ('is_fraud').")

    if df_labeled.empty:
        print("No records with ground truth labels found. Cannot calculate accuracy.")
        print("Make sure you ran the setup script to generate labeled mock data,")
        print("and processed it (e.g., using scripts/run_simulation.py or letting the backend run).") # <-- FIXED SCRIPT NAME
        return

    # Extract ground truth and predictions
    y_true = df_labeled['is_fraud'].astype(int) # Convert to standard int for sklearn
    y_pred = df_labeled['final_status'].map(STATUS_TO_PREDICTION).fillna(0).astype(int) # Map status to 0/1, default unknown to 0

    # Ensure we have both 0 and 1 in predictions if possible for full report
    labels = sorted(pd.unique(y_true.tolist() + y_pred.tolist()))
    target_names = ['Class 0 (Normal)', 'Class 1 (Fraud)']

    # Adjust target_names if only one class is present in y_true (unlikely but possible)
    if len(labels) == 1:
        if labels[0] == 0:
             target_names = ['Class 0 (Normal)']
        else:
             target_names = ['Class 1 (Fraud)']


    # --- Calculate Metrics ---
    print("\n--- Accuracy Metrics ---")

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nConfusion Matrix:")
    # Use labels parameter to ensure matrix dimensions are correct even if one class is missing
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)
    # Explain the confusion matrix structure
    if len(labels) == 2:
        print("  [[TN  FP]")
        print("   [FN  TP]]")
        print(f"  TN (True Negatives - Correctly Approved): {cm[0, 0]}")
        print(f"  FP (False Positives - Incorrectly Flagged/Denied): {cm[0, 1]}")
        print(f"  FN (False Negatives - Incorrectly Approved): {cm[1, 0]}")
        print(f"  TP (True Positives - Correctly Flagged/Denied): {cm[1, 1]}")
    elif len(labels) == 1 and labels[0] == 0:
         print(f"  TN (True Negatives - Correctly Approved): {cm[0, 0]}")
    elif len(labels) == 1 and labels[0] == 1:
         print(f"  TP (True Positives - Correctly Flagged/Denied): {cm[0, 0]}")


    print("\nClassification Report:")
    # Use zero_division=0 to avoid warnings if a class has no support
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print(report)

    print("=" * 60)
    print("Calculation Complete.")
    print("Note: These metrics are based on the SIMULATED ground truth in the mock data.")
    print("      'Fraud' (Class 1) includes predictions of DENY or FLAG_FOR_REVIEW.")
    print("=" * 60)


if __name__ == "__main__":
    calculate_accuracy()