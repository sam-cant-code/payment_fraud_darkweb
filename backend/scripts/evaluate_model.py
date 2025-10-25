"""
IMPROVED Model Evaluation Script (V4.0)
- Generates comprehensive classification report
- Plots ROC Curve, Precision-Recall Curve, and Confusion Matrix
- Can be run on the 'latest' model or a specific timestamp
"""

import json
import pickle
import os
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Add backend/scripts to sys.path to find data_and_features
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.append(backend_dir) # Add backend dir
sys.path.append(script_dir) # Add scripts dir

try:
    from data_and_features import (
        PROJECT_ROOT,
        create_enhanced_mock_transactions,
        engineer_features_v4
    )
except ImportError as e:
    print(f"Error: Could not import from data_and_features.py: {e}")
    print("Please ensure this script is run from within the 'scripts' directory or that the backend path is correctly set.")
    sys.exit(1)


def plot_roc_curve(y_test, y_pred_proba, save_path):
    """Plot ROC curve"""
    if len(np.unique(y_test)) < 2:
        print("   ! Skipping ROC curve: Only one class present in test set.")
        return
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_val = auc(fpr, tpr) # Renamed to avoid conflict
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curve saved: {os.path.basename(save_path)}")


def plot_precision_recall_curve(y_test, y_pred_proba, save_path):
    """Plot Precision-Recall curve"""
    if len(np.unique(y_test)) < 2:
        print("   ! Skipping PR curve: Only one class present in test set.")
        return
        
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba) # Renamed
    pr_auc_val = average_precision_score(y_test, y_pred_proba) # Renamed
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc_val:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.ylim([0.0, 1.05]) # Ensure y-axis starts at 0
    plt.xlim([0.0, 1.0])  # Ensure x-axis starts at 0
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ PR curve saved: {os.path.basename(save_path)}")


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Normal', 'Fraud']
    # Handle cases where cm might not be 2x2 if only one class predicted/present
    tick_marks = np.arange(len(classes))
    ax.set(xticks=tick_marks,
           yticks=tick_marks,
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2. if cm.size > 0 else 0.
    # Iterate based on actual cm shape
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {os.path.basename(save_path)}")


def evaluate_model_comprehensive(X, y, model_timestamp):
    """Comprehensive model evaluation with visualizations"""
    print("\n" + "="*70)
    print(f"COMPREHENSIVE EVALUATION: {model_timestamp}")
    print("="*70)

    # --- 1. Load Model Files ---
    model_path = PROJECT_ROOT / 'models' / f'fraud_model_{model_timestamp}.pkl'
    scaler_path = PROJECT_ROOT / 'models' / f'scaler_{model_timestamp}.pkl'
    metadata_path = PROJECT_ROOT / 'models' / f'model_metadata_{model_timestamp}.json'

    if not model_path.exists() or not scaler_path.exists():
        print(f"Error: Model files not found for '{model_timestamp}'")
        print(f"Checked: {model_path}")
        return

    model = None
    scaler = None
    metadata = {}
    optimal_threshold = 0.5 # Default

    try:
        with open(model_path, 'rb') as f:  
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:  
            scaler = pickle.load(f)
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                optimal_threshold = metadata.get('optimal_threshold', 0.5)
                print(f"✓ Model loaded: {metadata.get('model_type', 'Unknown')}")
                print(f"✓ Features: {metadata.get('n_features', 'Unknown')}")
                print(f"✓ Optimal Threshold from metadata: {optimal_threshold:.4f}")
        else:
            print(f"✓ Model and Scaler loaded (no metadata found).")
            print(f"  Using default 0.5 threshold for classification report.")
        
    except Exception as e:
        print(f"Error loading model/metadata: {e}")
        return

    # --- 2. Get Test Data ---
    # Apply the same 80/20 split used during training
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    if len(y_test) == 0:
        print("Error: No test data found. Aborting evaluation.")
        return
        
    num_fraud_test = sum(y_test)
    num_normal_test = len(y_test) - num_fraud_test
    print(f"\n✓ Test set: {len(X_test)} samples ({num_fraud_test} fraud, {num_normal_test} normal)")

    # --- 3. Scale Data and Predict ---
    print("\n✓ Scaling test data and making predictions...")
    try:
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    except Exception as e:
        print(f"Error during scaling or prediction: {e}")
        return
        
    # Use the optimal threshold from metadata for classification
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # --- 4. Print Metrics to Console ---
    print("\n" + "="*70)
    print("MODEL EVALUATION (on Final Test Set)")
    print(f"(Using Threshold = {optimal_threshold:.4f} for Classification Report & CM)")
    print("="*70)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"           Predicted")
    print(f"           Normal  Fraud")
    print(f"Actual Normal  {cm[0][0]:>6d}  {cm[0][1]:>6d}")
    print(f"       Fraud   {cm[1][0]:>6d}  {cm[1][1]:>6d}")
    
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, 0)

    # Calculate metrics - ROC/PR AUC use probabilities, others use thresholded predictions
    roc_auc_val = 0.0
    pr_auc_val = 0.0
    f1 = 0.0
    recall = 0.0
    precision = 0.0
    fp_rate = 0.0

    if num_fraud_test > 0 and num_normal_test > 0 : # Need both classes for AUCs
        roc_auc_val = roc_auc_score(y_test, y_pred_proba)
        pr_auc_val = average_precision_score(y_test, y_pred_proba)
    elif num_fraud_test == 0:
         print("\nWarning: No fraud samples in the test set. ROC/PR AUC cannot be calculated.")
    elif num_normal_test == 0:
         print("\nWarning: No normal samples in the test set. ROC/PR AUC cannot be calculated.")

    # Calculate threshold-dependent metrics
    if (tp + fp + fn + tn) > 0: # Ensure denominator is not zero
         if (tp + fn) > 0: recall = tp / (tp + fn)
         if (tp + fp) > 0: precision = tp / (tp + fp)
         if (2*tp + fp + fn) > 0: f1 = 2 * tp / (2 * tp + fp + fn)
         if (fp + tn) > 0: fp_rate = fp / (fp + tn)

    print(f"\n--- Key Metrics (Test Set) ---")
    print(f"ROC-AUC Score:      {roc_auc_val:.4f} (Probability-based)")
    print(f"PR-AUC Score:       {pr_auc_val:.4f} (Probability-based)")
    print(f"F1-Score (Fraud):    {f1:.4f} (at threshold {optimal_threshold:.4f})")
    print(f"Recall (Fraud):      {recall:.4f} (at threshold {optimal_threshold:.4f})")
    print(f"Precision (Fraud):   {precision:.4f} (at threshold {optimal_threshold:.4f})")
    print(f"False Positive Rate: {fp_rate:.4f} (at threshold {optimal_threshold:.4f})")

    # --- 5. Generate and Save Plots ---
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Create a dedicated output directory for this evaluation run
    output_dir = PROJECT_ROOT / 'output' / f'evaluation_{model_timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Saving plots to: {output_dir}")

    # Plotting requires both classes to be present in y_test
    if num_fraud_test > 0 and num_normal_test > 0:
        plot_roc_curve(y_test, y_pred_proba, output_dir / 'roc_curve.png')
        plot_precision_recall_curve(y_test, y_pred_proba, output_dir / 'precision_recall_curve.png')
    else:
         print("  ! Skipping ROC and PR curve plots as only one class is present in the test set.")
         
    plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')
    
    print("\n✓ Comprehensive evaluation complete.")


def main():
    """
    Main function to run the evaluation.
    Usage:
    - python evaluate_model.py
    (evaluates the 'latest' model)
    - python evaluate_model.py 20241025_153000
    (evaluates a specific model timestamp)
    """
    
    model_timestamp = None
    
    # Check for command-line argument
    if len(sys.argv) > 1:
        model_timestamp = sys.argv[1]
        print(f"Attempting to evaluate specific model: {model_timestamp}")
    else:
        # Try to find the latest model timestamp
        try:
            timestamp_file = PROJECT_ROOT / 'models' / 'latest_model_timestamp.txt'
            if timestamp_file.exists():
                with open(timestamp_file, 'r') as f:
                    model_timestamp = f.read().strip()
                print(f"Found latest model timestamp: {model_timestamp}")
            else:
                print("Error: 'latest_model_timestamp.txt' not found. Cannot determine which model to evaluate.")
                print("Please run 'initialize_and_train.py' first or provide a model timestamp as an argument.")
                return
        except Exception as e:
            print(f"Error reading latest timestamp file: {e}")
            return

    if not model_timestamp or not (PROJECT_ROOT / 'models' / f'fraud_model_{model_timestamp}.pkl').exists():
        print(f"Error: Model files for timestamp '{model_timestamp}' not found.")
        return

    # Load the full dataset (using load_if_exists=True)
    # This is necessary to re-apply the *exact same* 80/20 split as training
    print("\n[STEP 1/2] Loading transaction data (may take a moment)...")
    # Use load_if_exists=True to avoid regenerating data just for evaluation
    transactions, wallets = create_enhanced_mock_transactions(load_if_exists=True) 
    
    if not transactions:
        print("Error: No transactions found in 'data/mock_blockchain_transactions.json'.")
        print("Please run 'initialize_and_train.py' first to generate data.")
        return

    # Engineer features using the same function as training
    print("\n[STEP 2/2] Engineering features (using V4 logic)...")
    # We pass wallets['threat_list'] but it's ignored by engineer_features_v4
    X, y, _ = engineer_features_v4(transactions, wallets['threat_list']) 

    if X is None or len(X) == 0:
        print("Error: Feature engineering failed. No data to evaluate.")
        return

    # Run the comprehensive evaluation
    evaluate_model_comprehensive(X, y, model_timestamp)


if __name__ == "__main__":
    main()