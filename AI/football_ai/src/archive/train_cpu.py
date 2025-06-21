#!/usr/bin/env python3
"""
CPU-Optimized XGBoost Training for VirtualBox
Designed for i9-14900KF with 10 cores
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import time
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "combined_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_and_prepare_data():
    """Load and prepare data for training."""
    print("📊 Loading features...")
    df = pd.read_parquet(DATA_FILE)
    
    print(f"✅ Loaded {len(df)} matches")
    print(f"📈 Features: {len(df.columns)}")
    
    # Prepare features and target - only keep numeric columns
    X = df.drop(columns=["Result", "Date", "HomeTeam", "AwayTeam", "Result_Encoded"], errors="ignore")
    
    # Remove any remaining non-numeric columns
    text_columns = X.select_dtypes(include=['object']).columns.tolist()
    if text_columns:
        print(f"🔧 Removing text columns: {text_columns}")
        X = X.drop(columns=text_columns)
    
    # Use encoded result if available
    if "Result_Encoded" in df.columns:
        y = df["Result_Encoded"]
    else:
        result_mapping = {'H': 0, 'D': 1, 'A': 2}
        y = df["Result"].map(result_mapping)
    
    y = y.astype(int)
    
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def create_cpu_optimized_params():
    """Create CPU-optimized XGBoost parameters."""
    return {
        # Core settings - CPU only
        'device': 'cpu',
        'tree_method': 'hist',  # Fast CPU algorithm
        'n_jobs': 8,  # Use 8 of 10 cores (leave some for system)
        
        # Model parameters
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        
        # Performance
        'verbosity': 1,
        'random_state': 42
    }

def train_cpu_model(X_train, y_train, X_val, y_val):
    """Train CPU-optimized XGBoost model."""
    print("🚀 Starting CPU-optimized training...")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # CPU-optimized parameters
    params = create_cpu_optimized_params()
    
    print("⚙️  Training parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Training with early stopping
    start_time = time.time()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
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
    
    return accuracy, y_pred, y_pred_proba

def save_model_and_results(model, accuracy, feature_names):
    """Save the trained model and results."""
    
    # Save XGBoost model
    model_file = MODELS_DIR / "cpu_xgboost_model.json"
    model.save_model(str(model_file))
    print(f"💾 Model saved to: {model_file}")
    
    # Save model metadata
    metadata = {
        'model_type': 'XGBoost CPU-Optimized',
        'accuracy': float(accuracy),
        'features': list(feature_names),
        'feature_count': len(feature_names),
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': 'CPU (i9-14900KF)',
        'cores_used': 8
    }
    
    metadata_file = MODELS_DIR / "cpu_model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: {metadata_file}")

def main():
    """Main training function."""
    print("🏈 CPU-OPTIMIZED FOOTBALL PREDICTION TRAINING")
    print("=" * 60)
    print("🖥️  Device: CPU (Intel i9-14900KF)")
    print("⚡ Cores: 8/10 (optimized for VirtualBox)")
    print("=" * 60)
    
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
    model = train_cpu_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    accuracy, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Save model and results
    save_model_and_results(model, accuracy, X.columns)
    
    print("\n🎉 Training completed successfully!")
    print(f"🏆 Final accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 