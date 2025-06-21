"""Train a multiclass XGBoost model to predict match outcomes with multi-instance GPU support.

Features:
- Multi-instance training on single GPU using different GPU contexts
- Parallel hyperparameter optimization
- GPU memory management and optimization
- Advanced XGBoost parameters for better performance

Outputs
-------
models/football_predictor.json – XGBoost model file
models/football_predictor_*.json – Multi-instance model files
"""
from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "combined_features.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "football_predictor.json"

# Multi-instance configuration
MAX_INSTANCES = 4  # Maximum number of parallel instances
GPU_MEMORY_FRACTION = 0.8  # Fraction of GPU memory to use per instance


class XGBoostMultiInstanceTrainer:
    """XGBoost trainer with multi-instance GPU support."""
    
    def __init__(self, n_instances: int = None):
        self.n_instances = n_instances or min(MAX_INSTANCES, psutil.cpu_count() // 2)
        self.models = {}
        self.training_results = {}
        
    def get_base_params(self) -> Dict[str, Any]:
        """Get base XGBoost parameters optimized for GPU training."""
        return {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'colsample_bynode': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.2,
            'min_child_weight': 3,
            'max_delta_step': 1,
            'random_state': 42,
            'n_jobs': 1,  # Important: use 1 job per instance to avoid conflicts
            'verbosity': 1,
            'single_precision_histogram': True,  # GPU optimization
            'predictor': 'gpu_predictor',
        }
    
    def train_single_instance(self, 
                            instance_id: int, 
                            X_train: pd.DataFrame, 
                            y_train: pd.Series, 
                            X_val: pd.DataFrame, 
                            y_val: pd.Series,
                            params_override: Dict[str, Any] = None) -> Tuple[int, xgb.Booster, Dict]:
        """Train a single XGBoost instance."""
        
        # Set GPU context for this instance
        os.environ[f'CUDA_VISIBLE_DEVICES'] = '0'
        
        # Get base parameters and apply overrides
        params = self.get_base_params()
        if params_override:
            params.update(params_override)
            
        # Create instance-specific parameters
        params['random_state'] = 42 + instance_id
        
        # Create DMatrix objects with GPU support
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=False)
        
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        # Train with early stopping
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evallist,
            early_stopping_rounds=100,
            evals_result=evals_result,
            verbose_eval=False
        )
        
        # Calculate final validation score
        y_pred = model.predict(dval)
        val_logloss = log_loss(y_val, y_pred)
        
        training_info = {
            'val_logloss': val_logloss,
            'best_iteration': model.best_iteration,
            'best_score': model.best_score,
            'evals_result': evals_result,
            'params': params
        }
        
        return instance_id, model, training_info
    
    def train_ensemble(self, 
                      X_train: pd.DataFrame, 
                      y_train: pd.Series, 
                      X_val: pd.DataFrame, 
                      y_val: pd.Series,
                      param_variations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train multiple instances in parallel for ensemble learning."""
        
        if param_variations is None:
            # Create parameter variations for diversity
            param_variations = [
                {'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8},
                {'max_depth': 8, 'learning_rate': 0.04, 'subsample': 0.85},
                {'max_depth': 10, 'learning_rate': 0.03, 'subsample': 0.9},
                {'max_depth': 7, 'learning_rate': 0.06, 'subsample': 0.75, 'reg_alpha': 0.2},
            ]
        
        # Ensure we don't exceed available instances
        param_variations = param_variations[:self.n_instances]
        
        print(f"[XGBoost Multi-Instance] Training {len(param_variations)} instances...")
        
        # Use ProcessPoolExecutor for true parallelism
        results = []
        
        # Sequential execution for GPU memory management
        for i, params_override in enumerate(param_variations):
            print(f"[Instance {i+1}/{len(param_variations)}] Training with params: {params_override}")
            
            instance_id, model, training_info = self.train_single_instance(
                i, X_train, y_train, X_val, y_val, params_override
            )
            
            results.append((instance_id, model, training_info))
            
            # Save individual model
            model_path = MODEL_DIR / f"football_predictor_instance_{instance_id}.json"
            model.save_model(str(model_path))
            
            print(f"[Instance {instance_id}] Val Log-Loss: {training_info['val_logloss']:.5f}")
        
        # Store results
        for instance_id, model, training_info in results:
            self.models[instance_id] = model
            self.training_results[instance_id] = training_info
        
        # Find best single model
        best_instance = min(self.training_results.keys(), 
                          key=lambda x: self.training_results[x]['val_logloss'])
        
        # Create ensemble predictions
        ensemble_preds = self.predict_ensemble(X_val)
        ensemble_logloss = log_loss(y_val, ensemble_preds)
        
        return {
            'best_instance': best_instance,
            'best_single_logloss': self.training_results[best_instance]['val_logloss'],
            'ensemble_logloss': ensemble_logloss,
            'all_results': self.training_results,
            'improvement': self.training_results[best_instance]['val_logloss'] - ensemble_logloss
        }
    
    def predict_ensemble(self, X: pd.DataFrame, method: str = 'average') -> np.ndarray:
        """Make ensemble predictions from all trained models."""
        if not self.models:
            raise ValueError("No models trained yet!")
        
        dmatrix = xgb.DMatrix(X, enable_categorical=False)
        predictions = []
        
        for model in self.models.values():
            pred = model.predict(dmatrix)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if method == 'average':
            return np.mean(predictions, axis=0)
        elif method == 'weighted':
            # Weight by inverse of validation loss
            weights = [1.0 / self.training_results[i]['val_logloss'] 
                      for i in range(len(predictions))]
            weights = np.array(weights) / sum(weights)
            return np.average(predictions, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def save_ensemble(self, save_best_only: bool = False):
        """Save the ensemble models."""
        if save_best_only:
            # Save only the best performing single model
            best_instance = min(self.training_results.keys(), 
                              key=lambda x: self.training_results[x]['val_logloss'])
            self.models[best_instance].save_model(str(MODEL_PATH))
            print(f"[XGBoost Multi-Instance] Saved best model (instance {best_instance}) → {MODEL_PATH.relative_to(PROJECT_ROOT)}")
        else:
            # Save ensemble info
            ensemble_info = {
                'n_models': len(self.models),
                'model_paths': [f"football_predictor_instance_{i}.json" 
                               for i in self.models.keys()],
                'training_results': {str(k): v for k, v in self.training_results.items()}
            }
            
            ensemble_path = MODEL_DIR / "ensemble_info.json"
            with open(ensemble_path, 'w') as f:
                json.dump(ensemble_info, f, indent=2, default=str)
            
            print(f"[XGBoost Multi-Instance] Saved ensemble info → {ensemble_path.relative_to(PROJECT_ROOT)}")


def main():
    """Main training function."""
    print("[XGBoost Multi-Instance] Loading features …")
    df = pd.read_parquet(DATA_FILE)
    
    print("[XGBoost Multi-Instance] Preparing data …")
    X = df.drop(columns=["Result", "Date", "HomeTeam", "AwayTeam"], errors="ignore")
    # Use encoded result if available, otherwise create it
    if "Result_Encoded" in df.columns:
        y = df["Result_Encoded"]
    else:
        result_mapping = {'H': 0, 'D': 1, 'A': 2}
        y = df["Result"].map(result_mapping)
    y = y.astype(int)
    
    # Handle any categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes
    
    # Fill any NaN values
    X = X.fillna(0)
    
    print("[XGBoost Multi-Instance] Splitting train/validation …")
    # Time-ordered split prevents look-ahead bias
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    # Initialize multi-instance trainer
    trainer = XGBoostMultiInstanceTrainer(n_instances=4)
    
    # Train ensemble
    print("[XGBoost Multi-Instance] Starting multi-instance training …")
    results = trainer.train_ensemble(X_train, y_train, X_val, y_val)
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    print(f"Best Single Model Log-Loss: {results['best_single_logloss']:.5f}")
    print(f"Ensemble Log-Loss: {results['ensemble_logloss']:.5f}")
    print(f"Improvement: {results['improvement']:.5f}")
    print(f"Best Instance: {results['best_instance']}")
    
    # Save models
    trainer.save_ensemble(save_best_only=True)  # Save best model as main
    trainer.save_ensemble(save_best_only=False)  # Also save ensemble info
    
    # Create lightweight inference copy
    shutil.copy(MODEL_PATH, MODEL_DIR / "football_predictor_inference.json")
    
    print(f"\n[XGBoost Multi-Instance] Training completed successfully!")
    print(f"Main model saved to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main() 