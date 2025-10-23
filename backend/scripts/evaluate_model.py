"""
V3.3 - Model Evaluation Script (with Command-Line Argument)
This script's ONLY job is to:
1. Load the dataset
2. Load a SPECIFIC, EXISTING model passed as an argument
3. Run the test data through it and print the accuracy report
"""

import json
import pickle
import os
import numpy as np
import sys  # <-- Import the sys module
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from datetime import datetime

# Import shared functions from our utility file
from data_and_features import (
    PROJECT_ROOT,
    create_enhanced_mock_transactions,
    engineer_temporal_features
)

# --- !! MODEL_TO_TEST_TIMESTAMP is now passed via command line !! ---
# (The hardcoded variable has been removed)


def evaluate_existing_model(X, y, model_timestamp):
    """Loads a specific model and evaluates it on the test set."""
    print("\n" + "="*70)
    print(f"EVALUATING MODEL: {model_timestamp}")
    print("="*70)

    # --- 1. Load Model Files ---
    model_path = PROJECT_ROOT / 'models' / f'fraud_model_{model_timestamp}.pkl'
    scaler_path = PROJECT_ROOT / 'models' / f'scaler_{model_timestamp}.pkl'
    metadata_path = PROJECT_ROOT / 'models' / f'model_metadata_{model_timestamp}.json'

    if not model_path.exists() or not scaler_path.exists():
        print(f"Error: Model files not found for timestamp '{model_timestamp}'")
        print(f"Checked for: {model_path}")
        print(f"Checked for: {scaler_path}")
        return

    try:
        with open(model_path, 'rb') as f: model = pickle.load(f)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        
        optimal_threshold = 0.5
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                optimal_threshold = metadata.get('optimal_threshold', 0.5)
        print(f"✓ Model, Scaler, and Metadata loaded.")
        print(f"✓ Using Optimal Threshold: {optimal_threshold:.3f}")
        
    except Exception as e:
        print(f"Error loading model files: {e}")
        return

    # --- 2. Get Test Data ---
    # Re-create the *exact same* 70/30 temporal split
    split_idx = int(len(X) * 0.7)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    print(f"✓ Test set loaded: {len(X_test)} samples ({sum(y_test)} fraud)")

    # --- 3. Scale, Predict, and Evaluate ---
    X_test_scaled = scaler.transform(X_test)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    # Use the loaded optimal threshold for prediction
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    print("\n" + "="*70)
    print("MODEL EVALUATION (on Test Set)")
    print("="*70)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                Normal  Fraud")
    print(f"Actual Normal   {cm[0][0]:>6d}   {cm[0][1]:>6d}")
    print(f"       Fraud    {cm[1][0]:>6d}   {cm[1][1]:>6d}")
    tn, fp, fn, tp = cm.ravel()

    roc_auc = roc_auc_score(y_test, y_pred_proba) if sum(y_test) > 0 else 0.0
    pr_auc = average_precision_score(y_test, y_pred_proba) if sum(y_test) > 0 else 0.0
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    fp_rate = fp/(fp+tn) if (fp+tn)>0 else 0

    print(f"\n--- Key Metrics ---")
    print(f"ROC-AUC Score:  {roc_auc:.4f}")
    print(f"PR-AUC Score:   {pr_auc:.4f}")
    print(f"F1-Score (Fraud): {f1:.4f}")
    print(f"Recall (Fraud):   {recall:.4f} (% of fraud caught)")
    print(f"Precision (Fraud):{precision:.4f} (% of flags that were correct)")
    print(f"FP Rate:          {fp_rate:.4f} (% of normal txs flagged as fraud)")
    print("="*70)


def main():
    """Main evaluation function"""
    start_time = datetime.now()
    print("=" * 70)
    print("V3.3: MODEL EVALUATION SCRIPT")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Check for command-line argument ---
    if len(sys.argv) < 2:
        print("Error: No model timestamp provided.")
        print("\nUsage:   python evaluate_model.py <timestamp>")
        print("Example: python evaluate_model.py 20251023_115133")
        return
    
    model_timestamp_to_test = sys.argv[1]
    # --- End of change ---

    # Load data (will load from file if it exists)
    transactions, wallets = create_enhanced_mock_transactions(load_if_exists=True)

    # Engineer features
    X, y, _ = engineer_temporal_features(transactions, wallets['threat_list'])

    # Evaluate the specified model
    if len(X) > 0:
        evaluate_existing_model(X, y, model_timestamp_to_test)
    else:
        print("Error: No features were generated.")

    end_time = datetime.now()
    print(f"\n✓ EVALUATION COMPLETE! (Duration: {end_time - start_time})")
    print("=" * 70)

if __name__ == "__main__":
    main()