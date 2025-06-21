#!/usr/bin/env python3
"""
Realistic XGBoost Training - Prevents Overfitting
Optimized for VirtualBox VM with i9-14900KF
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import time
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "combined_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_and_prepare_data():
    """Load and prepare data for realistic training."""
    print("📊 Loading features...")
    df = pd.read_parquet(DATA_FILE)
    
    print(f"✅ Loaded {len(df)} matches")
    
    # Prepare features - exclude target and identifiers
    feature_cols = [col for col in df.columns if col not in [
        'Result', 'Result_Encoded', 'Date', 'HomeTeam', 'AwayTeam', 
        'League', 'Country', 'Season', 'Round', 'Source', 'MatchID'
    ]]
    
    X = df[feature_cols]
    y = df['Result_Encoded']
    
    print(f"📈 Features used: {len(X.columns)}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def create_realistic_params():
    """Create parameters that prevent overfitting."""
    return {
        # Core settings - CPU optimized
        'device': 'cpu',
        'tree_method': 'hist',
        'n_jobs': 8,
        
        # Model parameters - prevent overfitting
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 4,  # Reduced from 6
        'learning_rate': 0.05,  # Reduced from 0.1
        'subsample': 0.7,  # Reduced from 0.8
        'colsample_bytree': 0.7,  # Reduced from 0.8
        'colsample_bylevel': 0.7,
        'reg_alpha': 1.0,  # Increased regularization
        'reg_lambda': 2.0,  # Increased regularization
        'min_child_weight': 3,  # Added
        'gamma': 0.1,  # Added
        
        # Performance
        'verbosity': 1,
        'random_state': 42
    }

def train_realistic_model(X_train, y_train, X_val, y_val):
    """Train realistic XGBoost model with early stopping."""
    print("🚀 Starting realistic training...")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Realistic parameters
    params = create_realistic_params()
    
    print("⚙️  Training parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Training with aggressive early stopping
    start_time = time.time()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,  # Reduced from 1000
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,  # More aggressive early stopping
        verbose_eval=50
    )
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.1f} seconds")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print("📊 Evaluating model...")
    
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Detailed classification report
    target_names = ['Home Win', 'Draw', 'Away Win']
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\n📈 Classification Report:")
    print(report)
    
    # Show some sample predictions to verify variety
    print("\n🔍 Sample Predictions (showing variety):")
    for i in range(min(10, len(y_pred_proba))):
        probs = y_pred_proba[i]
        pred_class = np.argmax(probs)
        actual_class = y_test.iloc[i]
        
        print(f"   Sample {i+1}: Predicted {target_names[pred_class]} ({probs[pred_class]:.3f}), "
              f"Actual {target_names[actual_class]}")
    
    return accuracy, y_pred, y_pred_proba

def save_realistic_model(model, accuracy, feature_names):
    """Save the realistic trained model."""
    
    # Save XGBoost model
    model_file = MODELS_DIR / "realistic_xgboost_model.json"
    model.save_model(str(model_file))
    print(f"💾 Model saved to: {model_file}")
    
    # Save model metadata
    metadata = {
        'model_type': 'XGBoost Realistic (Anti-Overfitting)',
        'accuracy': float(accuracy),
        'features': list(feature_names),
        'feature_count': len(feature_names),
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': 'CPU (i9-14900KF)',
        'cores_used': 8,
        'regularization': 'High (prevents overfitting)',
        'early_stopping': True
    }
    
    metadata_file = MODELS_DIR / "realistic_model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: {metadata_file}")

def main():
    """Main realistic training function."""
    print("🏈 REALISTIC FOOTBALL PREDICTION TRAINING")
    print("=" * 70)
    print("🖥️  Device: CPU (Intel i9-14900KF)")
    print("⚡ Cores: 8/10 (VM optimized)")
    print("🛡️  Anti-Overfitting: High regularization + early stopping")
    print("=" * 70)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Split data
    print("🔀 Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"📊 Training set: {len(X_train)} matches")
    print(f"📊 Validation set: {len(X_val)} matches")
    print(f"📊 Test set: {len(X_test)} matches")
    
    # Train model
    model = train_realistic_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    accuracy, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Save model and results
    save_realistic_model(model, accuracy, X.columns)
    
    print(f"\n🎉 Realistic training completed!")
    print(f"🏆 Final accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"✅ Model should now give varied, realistic predictions")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 