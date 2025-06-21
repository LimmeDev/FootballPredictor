#!/usr/bin/env python3
"""
Advanced XGBoost Hyperparameter Tuning for Football Predictions
Uses Optuna for smart hyperparameter optimization with cross-validation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss
import optuna
from optuna.samplers import TPESampler
import joblib
from pathlib import Path
import time
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "advanced_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

class FootballModelTuner:
    def __init__(self, n_trials=100, cv_folds=5, n_jobs=8):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.study = None
        self.X = None
        self.y = None
        
    def load_data(self):
        """Load the advanced features dataset."""
        print("📊 Loading advanced features...")
        df = pd.read_parquet(DATA_FILE)
        
        # Prepare features and target
        exclude_cols = [
            'Result', 'Result_Encoded', 'Date', 'HomeTeam', 'AwayTeam', 
            'League', 'Country', 'Season', 'Round', 'Source', 'MatchID'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.X = df[feature_cols]
        self.y = df['Result_Encoded']
        
        print(f"✅ Loaded {len(df)} matches with {len(feature_cols)} features")
        print(f"🎯 Target distribution: {self.y.value_counts().to_dict()}")
        
        return True
    
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization."""
        
        # Define hyperparameter search space
        params = {
            'device': 'cpu',
            'tree_method': 'hist',
            'n_jobs': self.n_jobs,
            'objective': 'multi:softprob',
            'num_class': 3,
            'verbosity': 0,
            'random_state': 42,
            
            # Tunable hyperparameters
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        }
        
        # Create XGBoost classifier
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, self.X, self.y, 
            cv=cv, 
            scoring='neg_log_loss',  # Use log loss for better probability calibration
            n_jobs=1  # XGBoost already uses multiple cores
        )
        
        return -scores.mean()  # Return positive log loss (lower is better)
    
    def tune_hyperparameters(self):
        """Run hyperparameter optimization."""
        print("🔧 Starting hyperparameter tuning...")
        print(f"🎯 Trials: {self.n_trials}")
        print(f"📊 Cross-validation folds: {self.cv_folds}")
        print(f"⚡ CPU cores: {self.n_jobs}")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='football_xgboost_tuning'
        )
        
        # Run optimization
        start_time = time.time()
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        tuning_time = time.time() - start_time
        
        self.study = study
        
        print(f"⏱️  Tuning completed in {tuning_time:.1f} seconds")
        print(f"🏆 Best log loss: {study.best_value:.4f}")
        
        # Show best parameters
        print("\n🎯 Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        
        return study.best_params
    
    def evaluate_best_model(self, best_params):
        """Evaluate the best model with more detailed metrics."""
        print("\n📊 Evaluating best model...")
        
        # Create model with best parameters
        model = xgb.XGBClassifier(**{
            **best_params,
            'device': 'cpu',
            'tree_method': 'hist',
            'n_jobs': self.n_jobs,
            'objective': 'multi:softprob',
            'num_class': 3,
            'verbosity': 0,
            'random_state': 42
        })
        
        # More comprehensive evaluation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold for final eval
        
        # Multiple metrics
        log_loss_scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='neg_log_loss')
        accuracy_scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='accuracy')
        
        results = {
            'log_loss_mean': -log_loss_scores.mean(),
            'log_loss_std': log_loss_scores.std(),
            'accuracy_mean': accuracy_scores.mean(),
            'accuracy_std': accuracy_scores.std(),
            'cv_folds': 10
        }
        
        print(f"📈 Cross-validation results (10-fold):")
        print(f"   Log Loss: {results['log_loss_mean']:.4f} ± {results['log_loss_std']:.4f}")
        print(f"   Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f} ({results['accuracy_mean']*100:.1f}%)")
        
        return results
    
    def save_tuned_model(self, best_params, eval_results):
        """Save the tuned model and results."""
        print("💾 Saving tuned model...")
        
        # Train final model on all data
        final_model = xgb.XGBClassifier(**{
            **best_params,
            'device': 'cpu',
            'tree_method': 'hist',
            'n_jobs': self.n_jobs,
            'objective': 'multi:softprob',
            'num_class': 3,
            'verbosity': 0,
            'random_state': 42
        })
        
        final_model.fit(self.X, self.y)
        
        # Save model
        model_file = MODELS_DIR / "tuned_xgboost_model.pkl"
        joblib.dump(final_model, model_file)
        
        # Save metadata
        metadata = {
            'model_type': 'XGBoost Tuned (Optuna)',
            'best_params': best_params,
            'evaluation_results': eval_results,
            'features': list(self.X.columns),
            'feature_count': len(self.X.columns),
            'training_samples': len(self.X),
            'tuning_trials': self.n_trials,
            'cv_folds': self.cv_folds,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': 'CPU (i9-14900KF)',
            'cores_used': self.n_jobs
        }
        
        metadata_file = MODELS_DIR / "tuned_model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved to: {model_file}")
        print(f"📋 Metadata saved to: {metadata_file}")
        
        return final_model

def main():
    """Main tuning function."""
    print("🏈 ADVANCED FOOTBALL MODEL TUNING")
    print("=" * 70)
    print("🔧 Hyperparameter optimization using Optuna")
    print("📊 Advanced features with team strength, form, injuries, H2H")
    print("🎯 Optimizing for realistic predictions")
    print("=" * 70)
    
    # Initialize tuner
    tuner = FootballModelTuner(n_trials=50, cv_folds=5, n_jobs=8)  # Reduced trials for VM
    
    # Load data
    if not tuner.load_data():
        return False
    
    # Run hyperparameter tuning
    best_params = tuner.tune_hyperparameters()
    
    # Evaluate best model
    eval_results = tuner.evaluate_best_model(best_params)
    
    # Save tuned model
    final_model = tuner.save_tuned_model(best_params, eval_results)
    
    print("\n🎉 Advanced tuning completed!")
    print(f"🏆 Best accuracy: {eval_results['accuracy_mean']:.3f} ({eval_results['accuracy_mean']*100:.1f}%)")
    print(f"📊 Log loss: {eval_results['log_loss_mean']:.4f}")
    print("✅ Ready for realistic football predictions!")
    
    # Feature importance
    print("\n🔍 Top 10 Most Important Features:")
    feature_importance = final_model.feature_importances_
    feature_names = tuner.X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.3f}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 