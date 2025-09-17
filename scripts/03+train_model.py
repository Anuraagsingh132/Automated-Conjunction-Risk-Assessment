import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

# Import scikit-learn components
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

# Import the advanced model
import xgboost as xgb

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Helper function to plot a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low-Risk', 'High-Risk'], 
                yticklabels=['Low-Risk', 'High-Risk'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_feature_importance(model, feature_names):
    """Helper function to plot feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

def plot_pr_curve(y_true, y_pred_proba, title='Precision-Recall Curve'):
    """Helper function to plot the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label=f'PR AUC = {pr_auc:.2f}')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("--- Starting Phase 3: Model Development & Training ---")
    
    # --- 1. DATA LOADING & ROBUSTNESS CHECK ---
    try:
        df = pd.read_csv('conjunction_events_sgp4.csv')
    except FileNotFoundError:
        print("Error: `conjunction_events_sgp4.csv` not found. Please run the Phase 2 script first.", file=sys.stderr)
        exit()

    TARGET = 'is_high_risk'
    
    min_samples_required = 5
    num_high_risk = df[TARGET].value_counts().get(1, 0)
    
    if num_high_risk < min_samples_required:
        print(f"\n❌ CRITICAL ERROR: Insufficient Data.", file=sys.stderr)
        print(f"The dataset contains only {num_high_risk} high-risk events.", file=sys.stderr)
        print(f"At least {min_samples_required} are needed for a meaningful train/test split.", file=sys.stderr)
        print("Please re-run the Phase 2 script with more satellites or a longer duration.", file=sys.stderr)
        exit()
        
    # --- 2. FEATURE SELECTION & DATA SPLITTING ---
    FEATURES = df.drop(columns=[TARGET, 'tca_jd', 'id_A', 'id_B']).columns.tolist()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split successfully:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # --- 3. BASELINE MODEL (RANDOM FOREST) ---
    print("\n--- Training Baseline Model (Random Forest) ---")
    rf_baseline = RandomForestClassifier(
        random_state=42, 
        n_estimators=100, 
        class_weight='balanced',
        n_jobs=-1  # --- THIS IS THE CHANGE: Use all available CPU cores ---
    )
    rf_baseline.fit(X_train, y_train)
    y_pred_test_rf = rf_baseline.predict(X_test)
    
    print("\nBaseline Model Performance (on Test Set):")
    print(classification_report(y_test, y_pred_test_rf, target_names=['Low-Risk', 'High-Risk']))

    # --- 4. ADVANCED MODEL (XGBOOST) ---
    print("\n--- Training Advanced Model (XGBoost) ---")
    
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Calculated scale_pos_weight for class imbalance: {scale_pos_weight:.2f}")

    xgb_tuned = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1  # --- THIS IS THE CHANGE: Use all available CPU cores ---
    )
    
    xgb_tuned.fit(X_train, y_train)
    
    # --- 5. FINAL EVALUATION ON TEST SET ---
    print("\n--- Final Evaluation of XGBoost Model on the Held-Out Test Set ---")
    y_pred_test_xgb = xgb_tuned.predict(X_test)
    y_pred_proba_xgb = xgb_tuned.predict_proba(X_test)[:, 1]
    
    print("\nFinal XGBoost Model Performance (on Test Set):")
    print(classification_report(y_test, y_pred_test_xgb, target_names=['Low-Risk', 'High-Risk']))
    
    print("\nPlotting Confusion Matrix:")
    plot_confusion_matrix(y_test, y_pred_test_xgb, title='Final XGBoost Confusion Matrix (Test Set)')
    
    print("\nPlotting Precision-Recall Curve:")
    plot_pr_curve(y_test, y_pred_proba_xgb)

    print("\nPlotting Feature Importance:")
    plot_feature_importance(xgb_tuned, FEATURES)

    # --- 6. SAVE THE FINAL MODEL ---
    model_filename = 'conjunction_model.joblib'
    joblib.dump(xgb_tuned, model_filename)
    print(f"\n✅ Phase 3 Complete! Final model saved to '{model_filename}'")
