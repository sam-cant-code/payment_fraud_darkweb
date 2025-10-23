"""
V3.2 - Model Training Script
This script's ONLY job is to:
1. Load/generate data
2. Train a NEW model
3. Save the model, scaler, DB, and threat list
"""

import json
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
from datetime import datetime

# Import shared functions from our new utility file
from data_and_features import (
    PROJECT_ROOT,
    create_enhanced_mock_transactions,
    engineer_temporal_features,
    create_directories,
    create_wallet_profiles_db,
    create_threat_list_file
)

# ================== MODEL TRAINING ==================
def train_enhanced_model(X, y):
    """Train model with proper evaluation and no data leakage"""
    print("\n" + "="*70)
    print("V3 MODEL TRAINING (On Temporal Features)")
    print("="*70)

    if len(X) == 0:
        print("Error: No features generated.")
        return

    # Temporal split (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n✓ Temporal split (70/30):")
    print(f"  Training: {len(X_train)} samples ({sum(y_train)} fraud)")
    print(f"  Testing:  {len(X_test)} samples ({sum(y_test)} fraud)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE
    print("\n✓ Applying SMOTE (0.3 ratio)...")
    try:
        k_neighbors_smote = min(5, max(1, sum(y_train)-1))
        smote = SMOTE(random_state=42, sampling_strategy=0.3, k_neighbors=k_neighbors_smote)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"  After SMOTE: {len(X_train_balanced)} samples ({sum(y_train_balanced)} fraud)")
    except ValueError as e:
        print(f"  SMOTE failed (not enough samples: {sum(y_train)}): {e}. Training on original data.")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    # Train model
    print("\n✓ Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150, max_depth=15,
        min_samples_split=10, min_samples_leaf=4,
        class_weight={0: 1, 1: 5}, # Increased weight for fraud
        max_features='sqrt', max_samples=0.8,
        random_state=42, n_jobs=-1, oob_score=True
    )
    model.fit(X_train_balanced, y_train_balanced)
    print(f"✓ Training complete (OOB Score: {model.oob_score_:.4f})")

    # Evaluate
    print("\n" + "="*70)
    print("MODEL EVALUATION (on Test Set)")
    print("="*70)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

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

    # Find optimal threshold
    optimal_threshold = 0.5
    if sum(y_test) > 0:
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        print(f"\nOptimal threshold for max F1: {optimal_threshold:.3f}")

    # Feature importance
    print("\n--- Top 15 Most Important Features ---")
    feature_names = [
        'value_eth', 'gas_price', 'total_cost', 'from_threat', 'to_threat', # 0-4
        'sender_tx_count', 'sender_avg_value', 'value_deviation',           # 5-7
        'sender_unique_recipients', 'recipient_diversity',                  # 8-9
        'receiver_tx_count', 'receiver_total_received', 'receiver_unique_senders', # 10-12
        'hour', 'weekday', 'odd_hours',                                     # 13-15
        'avg_time_between_tx', 'min_time_between_tx',                      # 16-17
        'very_high_value', 'high_value', 'micro_value', 'high_gas', 'low_gas', # 18-22
        'gas_value_ratio', 'self_transfer', 'round_number',                   # 23-25
    ]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:<25s}: {importances[idx]:.4f}")

    # Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = PROJECT_ROOT / 'models' / f'fraud_model_{run_timestamp}.pkl'
    scaler_path = PROJECT_ROOT / 'models' / f'scaler_{run_timestamp}.pkl'
    metadata_path = PROJECT_ROOT / 'models' / f'model_metadata_{run_timestamp}.json'

    os.makedirs(model_path.parent, exist_ok=True)
    with open(model_path, 'wb') as f: pickle.dump(model, f)
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {scaler_path}")

    # Save metadata
    metadata = {
        'model_type': 'RandomForestClassifier', 'n_features': len(feature_names),
        'optimal_threshold': float(optimal_threshold),
        'test_f1': float(f1), 'test_recall': float(recall),
        'test_precision': float(precision), 'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc), 'fp_rate': float(fp_rate),
        'training_date': datetime.now().isoformat(),
        'class_weight': {0: 1, 1: 5},
        'features_used': feature_names,
        'model_file': os.path.basename(str(model_path)),
        'scaler_file': os.path.basename(str(scaler_path))
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")

    # Save latest timestamp
    timestamp_file = PROJECT_ROOT / 'models' / 'latest_model_timestamp.txt'
    with open(timestamp_file, 'w') as f: f.write(run_timestamp)
    print(f"✓ Latest timestamp saved to: {timestamp_file}")


def main():
    """Main setup function"""
    start_time = datetime.now()
    print("=" * 70)
    print("V3.2: MODEL TRAINING SCRIPT")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    create_directories()

    # Generate data (will not regenerate if file exists)
    transactions, wallets = create_enhanced_mock_transactions(load_if_exists=False)

    # Engineer features (temporally) and get profiles
    X, y, final_profiles = engineer_temporal_features(transactions, wallets['threat_list'])

    # Train model
    if len(X) > 0:
        train_enhanced_model(X, y)
    else:
        print("Error: No features were generated. Skipping model training.")

    # Create supporting files
    create_threat_list_file(wallets['threat_list'])
    create_wallet_profiles_db(wallets['all_wallets'], final_profiles)

    end_time = datetime.now()
    print("\n" + "=" * 70)
    print(f"✓ TRAINING COMPLETE! (Duration: {end_time - start_time})")
    print("=" * 70)

if __name__ == "__main__":
    main()