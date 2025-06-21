#!/usr/bin/env python3
"""
STREAMLINED MODEL TUNING
Single file for intensive model training with time control

Usage: python Tune_Model.py --20        (train for 20 minutes)
       python Tune_Model.py --5         (train for 5 minutes) 
       python Tune_Model.py --60        (train for 1 hour)
"""

import pandas as pd
import numpy as np
import json
import time
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "mega_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "training_logs"
LOGS_DIR.mkdir(exist_ok=True)

class LiveLogger:
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.start_time = time.time()
        
        # Setup file logging
        log_file = LOGS_DIR / f"training_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🚀 TRAINING SESSION STARTED: {session_name}")
        self.logger.info(f"📝 Log file: {log_file}")
    
    def log_phase(self, phase_name: str, phase_number: int, total_phases: int):
        """Log phase transition."""
        elapsed = (time.time() - self.start_time) / 60
        self.logger.info("=" * 80)
        self.logger.info(f"📊 PHASE {phase_number}/{total_phases}: {phase_name.upper()}")
        self.logger.info(f"⏱️  Elapsed: {elapsed:.1f} minutes")
        self.logger.info("=" * 80)
    
    def log_model_start(self, model_name: str, params: Dict = None):
        """Log model training start."""
        self.logger.info(f"🤖 Training {model_name}...")
        if params:
            self.logger.info(f"⚙️  Parameters: {params}")
    
    def log_model_result(self, model_name: str, scores: Dict, training_time: float = None):
        """Log model training results."""
        self.logger.info(f"📊 {model_name} Results:")
        for metric, score in scores.items():
            if isinstance(score, float):
                self.logger.info(f"   {metric}: {score:.4f}")
            else:
                self.logger.info(f"   {metric}: {score}")
        if training_time:
            self.logger.info(f"   Training time: {training_time:.2f}s")
    
    def log_hyperparameter_search(self, model_name: str, n_iter: int, param_grid: Dict):
        """Log hyperparameter search details."""
        self.logger.info(f"🔍 Hyperparameter Search: {model_name}")
        self.logger.info(f"🎯 Iterations: {n_iter}")
        self.logger.info(f"📋 Parameter Space:")
        for param, values in param_grid.items():
            if isinstance(values, list):
                self.logger.info(f"   {param}: {len(values)} options {values[:3]}{'...' if len(values) > 3 else ''}")
            else:
                self.logger.info(f"   {param}: {values}")
    
    def log_best_params(self, model_name: str, best_score: float, best_params: Dict):
        """Log best parameters found."""
        self.logger.info(f"🏆 Best {model_name} Configuration:")
        self.logger.info(f"   Score: {best_score:.4f}")
        self.logger.info(f"   Parameters:")
        for param, value in best_params.items():
            self.logger.info(f"     {param}: {value}")
    
    def log_progress(self, current: int, total: int, item_name: str = ""):
        """Log progress with visual progress bar."""
        percent = (current / total) * 100
        bar_length = 30
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        elapsed = (time.time() - self.start_time) / 60
        self.logger.info(f"⏳ Progress: [{bar}] {percent:.1f}% | {current}/{total} {item_name} | {elapsed:.1f}min elapsed")
    
    def log_final_summary(self, best_score: float, total_models: int, total_time: float):
        """Log final session summary."""
        self.logger.info("=" * 80)
        self.logger.info("🎉 TRAINING SESSION COMPLETE!")
        self.logger.info(f"🏆 Best Score Achieved: {best_score:.4f} ({best_score*100:.2f}%)")
        self.logger.info(f"🤖 Total Models Trained: {total_models}")
        self.logger.info(f"⏱️  Total Time: {total_time:.1f} minutes")
        self.logger.info("=" * 80)

class StreamlinedTuner:
    def __init__(self, minutes: int):
        self.target_duration = minutes * 60  # Convert to seconds
        self.start_time = time.time()
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.logger = LiveLogger(f"{minutes}-minute Training Session")
        self.model_counter = 0
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data."""
        self.logger.logger.info("📊 Loading mega features dataset...")
        df = pd.read_parquet(DATA_FILE)
        
        # Exclude data leakage features
        exclude_cols = [
            'Home_Goals', 'Away_Goals', 'Total_Goals', 'Goal_Difference',
            'Home_Win_Margin', 'Away_Win_Margin', 'Home_xG_Est', 'Away_xG_Est',
            'Result', 'Result_Encoded', 'Date', 'HomeTeam', 'AwayTeam', 
            'League', 'Country', 'Season', 'Round', 'Source', 'MatchID',
            'Home_Formation', 'Away_Formation', 'Weather'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df['Result_Encoded']
        
        self.logger.logger.info(f"✅ Dataset loaded successfully:")
        self.logger.logger.info(f"   📈 Matches: {len(df):,}")
        self.logger.logger.info(f"   🔢 Features: {len(X.columns)}")
        self.logger.logger.info(f"   🎯 Target distribution: {dict(y.value_counts())}")
        
        return X, y
    
    def time_remaining(self) -> float:
        """Get remaining time in seconds."""
        elapsed = time.time() - self.start_time
        return max(0, self.target_duration - elapsed)
    
    def train_quick_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train quick baseline models."""
        self.logger.log_phase("Quick Baseline Models", 1, 3)
        
        models = {
            'Decision Tree': {
                'model': DecisionTreeClassifier,
                'params': {'max_depth': 15, 'min_samples_split': 10, 'random_state': 42}
            },
            'Random Forest': {
                'model': RandomForestClassifier,
                'params': {'n_estimators': 100, 'max_depth': 10, 'n_jobs': 8, 'random_state': 42}
            },
            'XGBoost': {
                'model': xgb.XGBClassifier,
                'params': {'n_estimators': 100, 'max_depth': 6, 'n_jobs': 8, 'random_state': 42, 'eval_metric': 'mlogloss'}
            }
        }
        
        results = {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        for i, (name, config) in enumerate(models.items()):
            if self.time_remaining() < 30:
                self.logger.logger.warning(f"⚠️  Skipping {name} - insufficient time remaining")
                break
            
            self.model_counter += 1
            self.logger.log_progress(i, len(models), "baseline models")
            self.logger.log_model_start(name, config['params'])
            
            # Train model
            start_time = time.time()
            model = config['model'](**config['params'])
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=4)
            
            # Full training
            model.fit(X_train, y_train)
            test_score = model.score(X_test, y_test)
            training_time = time.time() - start_time
            
            scores = {
                'CV Score': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'Test Score': test_score
            }
            
            self.logger.log_model_result(name, scores, training_time)
            
            results[name] = {
                'model': model,
                'cv_score': cv_scores.mean(),
                'test_score': test_score
            }
            
            if test_score > self.best_score:
                self.best_score = test_score
                self.best_model = model
                self.logger.logger.info(f"🌟 NEW BEST MODEL: {name} with score {test_score:.4f}")
                
        return results
    
    def hyperparameter_search(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Intensive hyperparameter search."""
        self.logger.log_phase("Hyperparameter Optimization", 2, 3)
        
        # Parameter grids
        rf_params = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [8, 12, 16, 20, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', 0.5, 0.7]
        }
        
        xgb_params = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
        
        results = {}
        
        # Calculate search iterations based on time
        time_per_model = self.time_remaining() / 2
        iterations_per_model = max(10, min(50, int(time_per_model / 30)))
        
        # Random Forest search
        if self.time_remaining() > 60:
            self.model_counter += 1
            self.logger.log_hyperparameter_search("Random Forest", iterations_per_model, rf_params)
            
            rf_search = RandomizedSearchCV(
                RandomForestClassifier(n_jobs=8, random_state=42),
                rf_params,
                n_iter=iterations_per_model,
                cv=3,
                scoring='accuracy',
                n_jobs=2,
                random_state=42,
                verbose=0
            )
            
            start_time = time.time()
            rf_search.fit(X, y)
            search_time = time.time() - start_time
            
            self.logger.log_best_params("Random Forest", rf_search.best_score_, rf_search.best_params_)
            self.logger.logger.info(f"⏱️  Search completed in {search_time:.1f}s")
            
            results['Random Forest Tuned'] = {
                'best_score': rf_search.best_score_,
                'best_params': rf_search.best_params_,
                'model': rf_search.best_estimator_
            }
            
            if rf_search.best_score_ > self.best_score:
                self.best_score = rf_search.best_score_
                self.best_model = rf_search.best_estimator_
                self.logger.logger.info(f"🌟 NEW BEST MODEL: Tuned Random Forest with score {rf_search.best_score_:.4f}")
        
        # XGBoost search
        if self.time_remaining() > 60:
            self.model_counter += 1
            self.logger.log_hyperparameter_search("XGBoost", iterations_per_model, xgb_params)
            
            xgb_search = RandomizedSearchCV(
                xgb.XGBClassifier(n_jobs=8, random_state=42, eval_metric='mlogloss'),
                xgb_params,
                n_iter=iterations_per_model,
                cv=3,
                scoring='accuracy',
                n_jobs=2,
                random_state=42,
                verbose=0
            )
            
            start_time = time.time()
            xgb_search.fit(X, y)
            search_time = time.time() - start_time
            
            self.logger.log_best_params("XGBoost", xgb_search.best_score_, xgb_search.best_params_)
            self.logger.logger.info(f"⏱️  Search completed in {search_time:.1f}s")
            
            results['XGBoost Tuned'] = {
                'best_score': xgb_search.best_score_,
                'best_params': xgb_search.best_params_,
                'model': xgb_search.best_estimator_
            }
            
            if xgb_search.best_score_ > self.best_score:
                self.best_score = xgb_search.best_score_
                self.best_model = xgb_search.best_estimator_
                self.logger.logger.info(f"🌟 NEW BEST MODEL: Tuned XGBoost with score {xgb_search.best_score_:.4f}")
        
        return results
    
    def final_ensemble_training(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train final ensemble models with remaining time."""
        self.logger.log_phase("Final Ensemble Training", 3, 3)
        
        # Use remaining time for intensive ensemble
        remaining_minutes = self.time_remaining() / 60
        n_estimators = min(1000, max(100, int(remaining_minutes * 50)))
        
        self.logger.logger.info(f"🕒 Time remaining: {remaining_minutes:.1f} minutes")
        self.logger.logger.info(f"🌲 Training ensembles with {n_estimators} estimators")
        
        final_models = {
            'Mega Random Forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'n_jobs': 8,
                    'random_state': 42,
                    'oob_score': True
                }
            },
            'Mega XGBoost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': min(500, n_estimators//2),
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_jobs': 8,
                    'random_state': 42,
                    'eval_metric': 'mlogloss'
                }
            }
        }
        
        results = {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for i, (name, config) in enumerate(final_models.items()):
            if self.time_remaining() < 60:
                self.logger.logger.warning(f"⚠️  Skipping {name} - insufficient time remaining")
                break
            
            self.model_counter += 1
            self.logger.log_progress(i, len(final_models), "ensemble models")
            self.logger.log_model_start(name, config['params'])
            
            model = config['model'](**config['params'])
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            test_score = model.score(X_test, y_test)
            
            scores = {'Test Score': test_score, 'Estimators': config['params']['n_estimators']}
            
            if hasattr(model, 'oob_score_'):
                scores['OOB Score'] = model.oob_score_
            
            self.logger.log_model_result(name, scores, training_time)
            
            results[name] = {
                'model': model,
                'test_score': test_score,
                'training_time': training_time,
                'n_estimators': config['params']['n_estimators']
            }
            
            if hasattr(model, 'oob_score_'):
                results[name]['oob_score'] = model.oob_score_
            
            if test_score > self.best_score:
                self.best_score = test_score
                self.best_model = model
                self.logger.logger.info(f"🌟 NEW BEST MODEL: {name} with score {test_score:.4f}")
        
        return results
    
    def save_best_model(self):
        """Save the best model found."""
        if self.best_model is not None:
            model_file = MODELS_DIR / "best_tuned_model.pkl"
            joblib.dump(self.best_model, model_file)
            self.logger.logger.info(f"💾 Best model saved to: {model_file}")
            self.logger.logger.info(f"🏆 Final best score: {self.best_score:.4f} ({self.best_score*100:.2f}%)")
        else:
            self.logger.logger.warning("⚠️  No model was successfully trained!")
    
    def run_tuning_session(self):
        """Run the complete tuning session."""
        self.logger.logger.info("🎯 STREAMLINED MODEL TUNING INITIATED")
        self.logger.logger.info(f"⏱️  Target Duration: {self.target_duration//60} minutes")
        self.logger.logger.info(f"🔥 Intensive hyperparameter optimization mode")
        
        # Load data
        X, y = self.load_data()
        
        # Phase 1: Quick baselines
        if self.time_remaining() > 0:
            self.results['baselines'] = self.train_quick_models(X, y)
        
        # Phase 2: Hyperparameter search
        if self.time_remaining() > 120:  # At least 2 minutes left
            self.results['tuned'] = self.hyperparameter_search(X, y)
        
        # Phase 3: Final ensemble
        if self.time_remaining() > 60:  # At least 1 minute left
            self.results['ensemble'] = self.final_ensemble_training(X, y)
        
        # Save best model
        self.save_best_model()
        
        # Final summary
        total_time = (time.time() - self.start_time) / 60
        self.logger.log_final_summary(self.best_score, self.model_counter, total_time)

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Streamlined Model Tuning')
    parser.add_argument('--minutes', type=int, default=20, help='Training duration in minutes')
    
    # Also support the --20 format by parsing sys.argv directly
    import sys
    if len(sys.argv) == 2 and sys.argv[1].startswith('--') and sys.argv[1][2:].isdigit():
        minutes = int(sys.argv[1][2:])
    else:
        args = parser.parse_args()
        minutes = args.minutes
    
    if minutes < 1 or minutes > 1440:  # 1 minute to 24 hours
        print("❌ Duration must be between 1 and 1440 minutes")
        return False
    
    # Run tuning
    tuner = StreamlinedTuner(minutes)
    tuner.run_tuning_session()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 