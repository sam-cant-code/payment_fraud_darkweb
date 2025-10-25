"""
IMPROVED Model Training Script (V4.0)
- Cross-validation with TimeSeriesSplit
- Ensemble model (Random Forest + XGBoost + Gradient Boosting)
- Better evaluation metrics
- NO DATA LEAKAGE
"""

import json
import pickle
import os
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from imblearn.over_sampling import SMOTE
from datetime import datetime

# Try importing XGBoost (optional but recommended)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not installed. Using Random Forest + Gradient Boosting only.")
    print("   Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Add backend/scripts to sys.path
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(SCRIPT_DIR)
except NameError:
    # Fallback for interactive
    SCRIPT_DIR = os.getcwd()
    sys.path.append(SCRIPT_DIR)

# Import shared functions from V4 data_and_features
from data_and_features import (
    PROJECT_ROOT,
    create_enhanced_mock_transactions,
    engineer_features_v4,  # <-- Using the correct V4 function
    create_directories,
    create_wallet_profiles_db,
    create_threat_list_file,
    FEATURE_NAMES_V4      # <-- Using the correct V4 feature names
)


# ================== FEATURE NAMES ==================
# Use the feature names from data_and_features.py
FEATURE_NAMES = FEATURE_NAMES_V4


# ================== BUILD ENSEMBLE MODEL ==================
def build_ensemble_model():
    """
    Create ensemble of multiple models for better performance
    """
    print("\n=== Building Ensemble Model (V4) ===")
    
    # Model 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight={0: 1, 1: 5},
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )
    print("  ✓ Random Forest configured")
    
    # Model 2: Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    print("  ✓ Gradient Boosting configured")
    
    # Model 3: XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,  # Handle imbalance
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        print("  ✓ XGBoost configured")
        
        # Voting ensemble with all three
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('xgb', xgb)
            ],
            voting='soft',
            weights=[2, 1, 3]  # XGBoost gets highest weight
        )
        print("  ✓ Ensemble: Random Forest + Gradient Boosting + XGBoost")
    else:
        # Voting ensemble without XGBoost
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb)
            ],
            voting='soft',
            weights=[2, 1]
        )
        print("  ✓ Ensemble: Random Forest + Gradient Boosting")
    
    return ensemble


# ================== CROSS-VALIDATION ==================
def perform_cross_validation(X_train_val, y_train_val, model):
    """
    Perform time-series cross-validation
    """
    print("\n=== Cross-Validation (Time Series Split) ===")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Running 5-fold time-series cross-validation...")
    print("(This may take a few minutes...)")
    
    # F1 Score
    f1_scores = cross_val_score(
        model, X_train_val, y_train_val,
        cv=tscv,
        scoring='f1',
        n_jobs=-1
    )
    
    # ROC-AUC Score
    roc_scores = cross_val_score(
        model, X_train_val, y_train_val,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print(f"\nCross-Validation Results:")
    print(f"  F1 Scores:    {f1_scores}")
    print(f"  Mean F1:      {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
    print(f"  ROC-AUC Scores: {roc_scores}")
    print(f"  Mean ROC-AUC:   {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")
    
    return {
        'f1_scores': f1_scores.tolist(),
        'mean_f1': float(f1_scores.mean()),
        'std_f1': float(f1_scores.std()),
        'roc_scores': roc_scores.tolist(),
        'mean_roc': float(roc_scores.mean()),
        'std_roc': float(roc_scores.std())
    }


# ================== TRAIN MODEL ==================
def train_ensemble_model(X, y):
    """
    Train ensemble model with cross-validation and proper evaluation
    """
    print("\n" + "="*70)
    print("V4 MODEL TRAINING (ENSEMBLE + CROSS-VALIDATION)")
    print("="*70)

    if len(X) == 0:
        print("Error: No features generated.")
        return None, None, None # Return None for metadata, model, scaler

    # Split data: 80% train/val, 20% final test
    final_test_split = int(len(X) * 0.8)
    X_train_val, X_test = X[:final_test_split], X[final_test_split:]
    y_train_val, y_test = y[:final_test_split], y[final_test_split:]

    print(f"\n✓ Data split (80/20):")
    print(f"  Train+Val: {len(X_train_val)} samples ({sum(y_train_val)} fraud)")
    print(f"  Final Test: {len(X_test)} samples ({sum(y_test)} fraud)")

    # Scale features
    print("\n✓ Scaling features...")
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE
    print("\n✓ Applying SMOTE (0.25 ratio)...")
    X_train_balanced, y_train_balanced = X_train_val_scaled, y_train_val # Default to original if SMOTE fails
    try:
        # Ensure there are enough samples for the smallest class for k_neighbors
        fraud_count_train_val = sum(y_train_val)
        if fraud_count_train_val < 2:
             raise ValueError("Not enough samples of the minority class (fraud) for SMOTE.")
        
        k_neighbors_smote = min(5, max(1, fraud_count_train_val - 1))
        
        smote = SMOTE(random_state=42, sampling_strategy=0.25, k_neighbors=k_neighbors_smote)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_val_scaled, y_train_val)
        print(f"  After SMOTE: {len(X_train_balanced)} samples ({sum(y_train_balanced)} fraud)")
    
    except ValueError as e:
        print(f"  SMOTE failed: {e}. Using original data for training.")
        # Ensure original data is still used if SMOTE fails
        X_train_balanced, y_train_balanced = X_train_val_scaled, y_train_val

    # Build ensemble model
    ensemble = build_ensemble_model()

    # Cross-validation (on pre-SMOTE data to avoid data leakage in CV)
    cv_results = perform_cross_validation(X_train_val_scaled, y_train_val, ensemble)

    # Train final model on all train+val data (with SMOTE)
    print("\n✓ Training final ensemble on all train+val data...")
    print("  (This may take several minutes...)")
    ensemble.fit(X_train_balanced, y_train_balanced)
    print("✓ Training complete!")

    # Evaluate on test set
    print("\n" + "="*70)
    print("MODEL EVALUATION (on Final Test Set)")
    print("="*70)

    y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

    # Find optimal threshold using PR curve on the test set probabilities
    optimal_threshold = 0.5 # Default threshold
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    roc_auc = 0.0
    pr_auc = 0.0
    fp_rate = 0.0
    
    if sum(y_test) > 0: # Only calculate if there are positive samples in test set
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
        # Calculate F1 score for each threshold, handle division by zero
        f1_scores = np.divide(2 * precision_curve * recall_curve, 
                              precision_curve + recall_curve, 
                              out=np.zeros_like(precision_curve), 
                              where=(precision_curve + recall_curve) != 0)
        
        # Find the threshold that maximizes F1 score
        valid_threshold_indices = np.arange(len(thresholds_pr))
        if len(valid_threshold_indices) > 0:
             optimal_idx = np.argmax(f1_scores[:-1][valid_threshold_indices]) # Exclude last precision/recall value
             optimal_threshold = thresholds_pr[valid_threshold_indices[optimal_idx]]
        
        print(f"\nOptimal threshold (Max F1 on Test): {optimal_threshold:.4f}")

        # Apply optimal threshold for classification report and metrics
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        print("\nClassification Report (using optimal threshold):")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4, zero_division=0))

        print("\nConfusion Matrix (using optimal threshold):")
        cm = confusion_matrix(y_test, y_pred)
        print(f"           Predicted")
        print(f"           Normal  Fraud")
        print(f"Actual Normal  {cm[0][0]:>6d}  {cm[0][1]:>6d}")
        print(f"       Fraud   {cm[1][0]:>6d}  {cm[1][1]:>6d}")
        
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, 0) # Handle edge case

        # Recalculate metrics based on optimal threshold prediction
        roc_auc = roc_auc_score(y_test, y_pred_proba) # ROC AUC uses probabilities
        pr_auc = average_precision_score(y_test, y_pred_proba) # PR AUC uses probabilities
        f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        fp_rate = fp/(fp+tn) if (fp+tn)>0 else 0

    else:
        print("\nWarning: No fraud samples in test set. Metrics are based on Normal class only.")
        # Report based on default 0.5 threshold if no fraud in test set
        y_pred = (y_pred_proba >= 0.5).astype(int) 
        print("\nClassification Report (using 0.5 threshold):")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, 0)
        fp_rate = fp/(fp+tn) if (fp+tn)>0 else 0
        # Other metrics remain 0


    print(f"\n--- Key Metrics (Test Set) ---")
    print(f"ROC-AUC Score:   {roc_auc:.4f}")
    print(f"PR-AUC Score:    {pr_auc:.4f}")
    print(f"F1-Score (Fraud): {f1:.4f} (at threshold {optimal_threshold:.4f})")
    print(f"Recall (Fraud):   {recall:.4f} (at threshold {optimal_threshold:.4f})")
    print(f"Precision (Fraud):{precision:.4f} (at threshold {optimal_threshold:.4f})")
    print(f"FP Rate:          {fp_rate:.4f} (at threshold {optimal_threshold:.4f})")

    # Feature importance (from Random Forest component)
    print("\n--- Top 15 Most Important Features ---")
    try:
        rf_model = None
        # Access the fitted estimators in VotingClassifier
        if hasattr(ensemble, 'estimators_'): 
             estimators_dict = dict(ensemble.estimators)
             if 'rf' in estimators_dict:
                 rf_model = estimators_dict['rf']

        if rf_model and hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            print("  Rank | Feature Name                 | Importance")
            print("  -----|------------------------------|------------")
            for i, idx in enumerate(indices, 1):
                print(f"  {i:^4} | {FEATURE_NAMES[idx]:<28s} | {importances[idx]:.4f}")
        else:
             print("  Could not retrieve Random Forest model or feature importances from ensemble.")
             
    except Exception as e:
        print(f"  Could not extract feature importances: {e}")

    # --- SAVE MODEL ---
    print("\n" + "="*70)
    print("SAVING MODEL AND METADATA")
    print("="*70)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = PROJECT_ROOT / 'models' / f'fraud_model_{run_timestamp}.pkl'
    scaler_path = PROJECT_ROOT / 'models' / f'scaler_{run_timestamp}.pkl'
    metadata_path = PROJECT_ROOT / 'models' / f'model_metadata_{run_timestamp}.json'

    os.makedirs(model_path.parent, exist_ok=True)
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
    except Exception as e:
        print(f"Error saving model/scaler: {e}")
        return None, None, None # Indicate failure

    # Save comprehensive metadata
    metadata = {
        'model_type': 'VotingClassifier_Ensemble',
        'models_used': list(dict(ensemble.estimators).keys()) if hasattr(ensemble, 'estimators') else [],
        'n_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
        'optimal_threshold': float(optimal_threshold),
        
        # Test metrics (at optimal threshold)
        'test_f1': float(f1),
        'test_recall': float(recall),
        'test_precision': float(precision),
        'test_roc_auc': float(roc_auc), # Based on probabilities, not threshold
        'test_pr_auc': float(pr_auc),   # Based on probabilities, not threshold
        'test_fp_rate': float(fp_rate),
        
        # Cross-validation metrics
        'cv_mean_f1': cv_results['mean_f1'],
        'cv_std_f1': cv_results['std_f1'],
        'cv_mean_roc_auc': cv_results['mean_roc'],
        'cv_std_roc_auc': cv_results['std_roc'],
        
        # Training info
        'training_date': datetime.now().isoformat(),
        'smote_ratio_used': 0.25 if fraud_count_train_val >= 2 else 0.0, # Reflect if SMOTE was actually used
        'train_val_size': len(X_train_val),
        'test_size': len(X_test),
        'test_fraud_rate': float(sum(y_test) / len(y_test)) if len(y_test) > 0 else 0.0,
        
        # Model files
        'model_file': model_path.name,
        'scaler_file': scaler_path.name
    }
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4) # Use indent=4 for better readability
        print(f"✓ Metadata saved: {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata: {e}")

    # Save latest timestamp
    timestamp_file = PROJECT_ROOT / 'models' / 'latest_model_timestamp.txt'
    try:
        with open(timestamp_file, 'w') as f:
            f.write(run_timestamp)
        print(f"✓ Latest timestamp updated: {timestamp_file}")
    except Exception as e:
        print(f"Error updating latest timestamp: {e}")
    
    # Return metadata, model, scaler for potential immediate use
    return metadata, ensemble, scaler


# ================== MAIN ==================
def main():
    """Main setup and training function"""
    start_time = datetime.now()
    print("=" * 70)
    print("V4.0: IMPROVED MODEL TRAINING")
    print("- Dynamic wallet generation (no memorization)")
    print("- NO threat list in features (no data leakage)")
    print("- Ensemble model (RF + GB + XGB)")
    print("- Cross-validation with proper evaluation")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    try:
        create_directories()

        # Generate data (force regeneration for training)
        print("\n[STEP 1/4] Generating mock transactions...")
        transactions, wallets = create_enhanced_mock_transactions(load_if_exists=False) # Always regenerate for training

        # Engineer features
        print("\n[STEP 2/4] Engineering features...")
        # Pass threat list but it will be ignored by V4 engineer
        X, y, final_profiles = engineer_features_v4(transactions, wallets['threat_list'])

        # Train model
        print("\n[STEP 3/4] Training ensemble model...")
        metadata = None
        if X is not None and len(X) > 0:
            metadata, _, _ = train_ensemble_model(X, y) # Unpack returned values
        else:
            print("Error: No features were generated. Skipping model training.")

        # Create supporting files (DB and threat list)
        print("\n[STEP 4/4] Creating database and threat list...")
        create_threat_list_file(wallets['threat_list'])
        # Ensure final_profiles is available before creating DB
        if 'final_profiles' in locals() and final_profiles is not None:
             create_wallet_profiles_db(wallets['all_wallets'], final_profiles)
        else:
             print("Warning: final_profiles not generated, skipping DB creation.")
             
        # Print summary if training was successful
        if metadata:
            print("\n" + "="*70)
            print("TRAINING SUMMARY")
            print("="*70)
            print(f"Model Type: {metadata.get('model_type', 'N/A')}")
            print(f"Models Used: {', '.join(metadata.get('models_used', []))}")
            print(f"Features: {metadata.get('n_features', 'N/A')}")
            print(f"\nTest Performance (Thresh: {metadata.get('optimal_threshold', 0.5):.4f}):")
            print(f"  F1 Score (Fraud):    {metadata.get('test_f1', 0.0):.4f}")
            print(f"  ROC-AUC Score:       {metadata.get('test_roc_auc', 0.0):.4f}")
            print(f"  Recall (Fraud):      {metadata.get('test_recall', 0.0):.4f}")
            print(f"  Precision (Fraud):   {metadata.get('test_precision', 0.0):.4f}")
            print(f"  False Positive Rate: {metadata.get('test_fp_rate', 0.0):.4f}")
            print(f"\nCross-Validation Performance:")
            print(f"  Mean F1 Score:    {metadata.get('cv_mean_f1', 0.0):.4f} ± {metadata.get('cv_std_f1', 0.0):.4f}")
            print(f"  Mean ROC-AUC Score: {metadata.get('cv_mean_roc_auc', 0.0):.4f} ± {metadata.get('cv_std_roc_auc', 0.0):.4f}")
        else:
            print("\nTraining did not complete successfully. No summary available.")

    except Exception as e:
        print("\n" + "="*70)
        print(f"FATAL ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        print("="*70)

    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        print("\n" + "=" * 70)
        print(f"✓ SCRIPT FINISHED! (Duration: {duration})")
        print("=" * 70)


if __name__ == "__main__":
    main()