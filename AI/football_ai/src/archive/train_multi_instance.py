"""Advanced multi-instance XGBoost training with GPU optimization and monitoring.

This script demonstrates the full capabilities of the XGBoost multi-instance GPU training system:
- Automatic GPU resource management
- Multi-instance ensemble training
- Real-time performance monitoring
- Advanced parameter optimization
- Model comparison and selection

Usage:
    python train_multi_instance.py --instances 4 --monitor
    python train_multi_instance.py --benchmark --tune-params
    python train_multi_instance.py --ensemble-only --instances 6
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from gpu_utils import GPUResourceManager, XGBoostGPUTrainer, benchmark_gpu_performance
from train_model import XGBoostMultiInstanceTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "training_features.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = MODEL_DIR / "multi_instance_results"
RESULTS_DIR.mkdir(exist_ok=True)


class AdvancedMultiInstanceTrainer:
    """Advanced trainer with comprehensive multi-instance capabilities."""
    
    def __init__(self, n_instances: int = None, enable_monitoring: bool = True):
        self.gpu_manager = GPUResourceManager()
        self.n_instances = n_instances or self.gpu_manager.get_optimal_instances()
        self.enable_monitoring = enable_monitoring
        
        self.training_history = []
        self.performance_metrics = {}
        self.ensemble_results = {}
        
        print(f"[Advanced Trainer] Initialized with {self.n_instances} instances")
        print(f"[Advanced Trainer] GPU Available: {self.gpu_manager.gpu_available}")
    
    def load_and_prepare_data(self) -> tuple:
        """Load and prepare data for training."""
        print("[Advanced Trainer] Loading data...")
        df = pd.read_parquet(DATA_FILE)
        
        X = df.drop(columns=["Result", "Date", "HomeTeam", "AwayTeam"], errors="ignore")
        y = df["Result"].astype(int)
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category').cat.codes
        
        # Fill NaN values
        X = X.fillna(0)
        
        print(f"[Advanced Trainer] Data shape: {X.shape}")
        print(f"[Advanced Trainer] Classes: {np.unique(y)}")
        
        return X, y
    
    def create_parameter_variations(self, base_params: Dict = None) -> List[Dict]:
        """Create diverse parameter variations for multi-instance training."""
        if base_params is None:
            base_params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.2,
                'min_child_weight': 3,
            }
        
        variations = []
        
        # Strategy 1: Different tree depths
        for depth in [6, 8, 10, 12]:
            params = base_params.copy()
            params.update({
                'max_depth': depth,
                'learning_rate': 0.05 + (depth - 8) * 0.005,  # Adjust LR with depth
            })
            variations.append(params)
        
        # Strategy 2: Different regularization strengths
        for alpha, lambda_reg in [(0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.3, 0.6)]:
            params = base_params.copy()
            params.update({
                'reg_alpha': alpha,
                'reg_lambda': lambda_reg,
                'learning_rate': 0.04 + alpha * 0.1,  # Adjust LR with regularization
            })
            variations.append(params)
        
        # Strategy 3: Different sampling strategies
        for subsample, colsample in [(0.7, 0.7), (0.8, 0.8), (0.9, 0.9), (0.85, 0.75)]:
            params = base_params.copy()
            params.update({
                'subsample': subsample,
                'colsample_bytree': colsample,
                'colsample_bylevel': colsample * 0.9,
                'colsample_bynode': colsample * 0.8,
            })
            variations.append(params)
        
        # Return only the number we need
        return variations[:self.n_instances]
    
    def train_single_instance_advanced(self, 
                                     instance_config: Dict,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     X_val: pd.DataFrame,
                                     y_val: pd.Series) -> Dict:
        """Train a single instance with advanced monitoring."""
        
        instance_id = instance_config['instance_id']
        params = instance_config['params']
        
        start_time = time.time()
        
        try:
            # Initialize GPU trainer
            trainer = XGBoostGPUTrainer(self.gpu_manager)
            
            # Create optimized DMatrix
            dtrain = trainer.create_optimized_dmatrix(X_train, y_train)
            dval = trainer.create_optimized_dmatrix(X_val, y_val)
            
            # Train with GPU optimization
            model = trainer.train_with_gpu_optimization(
                params,
                dtrain,
                num_boost_round=2000,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                early_stopping_rounds=100,
                instance_id=instance_id
            )
            
            # Get predictions and calculate metrics
            y_pred = model.predict(dval)
            from sklearn.metrics import log_loss, accuracy_score
            
            val_logloss = log_loss(y_val, y_pred)
            val_accuracy = accuracy_score(y_val, np.argmax(y_pred, axis=1))
            
            training_time = time.time() - start_time
            
            # Save model
            model_path = MODEL_DIR / f"multi_instance_{instance_id}.json"
            model.save_model(str(model_path))
            
            result = {
                'instance_id': instance_id,
                'model_path': str(model_path),
                'params': params,
                'val_logloss': val_logloss,
                'val_accuracy': val_accuracy,
                'training_time': training_time,
                'best_iteration': model.best_iteration,
                'status': 'success'
            }
            
            print(f"[Instance {instance_id}] Completed - Loss: {val_logloss:.5f}, "
                  f"Accuracy: {val_accuracy:.3f}, Time: {training_time:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"[Instance {instance_id}] Failed: {e}")
            return {
                'instance_id': instance_id,
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time
            }
    
    def monitor_training_progress(self) -> Dict:
        """Monitor GPU usage and training progress."""
        if not self.enable_monitoring:
            return {}
        
        gpu_stats = self.gpu_manager.monitor_gpu_usage()
        
        return {
            'timestamp': time.time(),
            'gpu_stats': gpu_stats,
            'active_instances': len(self.gpu_manager.active_instances),
        }
    
    def train_ensemble_advanced(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train ensemble with advanced multi-instance coordination."""
        
        print(f"[Advanced Trainer] Starting ensemble training with {self.n_instances} instances")
        
        # Create parameter variations
        param_variations = self.create_parameter_variations()
        
        # Prepare instance configurations
        instance_configs = []
        for i, params in enumerate(param_variations):
            config = {
                'instance_id': i,
                'params': params,
                'strategy': f"variation_{i}"
            }
            instance_configs.append(config)
        
        # Train instances
        results = []
        
        if self.gpu_manager.gpu_available:
            # Sequential training for GPU memory management
            print("[Advanced Trainer] Training instances sequentially for optimal GPU usage...")
            
            for config in tqdm(instance_configs, desc="Training instances"):
                # Monitor before training
                if self.enable_monitoring:
                    monitor_data = self.monitor_training_progress()
                    self.training_history.append(monitor_data)
                
                # Train instance
                result = self.train_single_instance_advanced(
                    config, X_train, y_train, X_val, y_val
                )
                results.append(result)
                
                # Small delay for GPU memory cleanup
                time.sleep(0.5)
        else:
            # Parallel training for CPU
            print("[Advanced Trainer] Training instances in parallel (CPU mode)...")
            
            with ThreadPoolExecutor(max_workers=min(4, self.n_instances)) as executor:
                futures = []
                for config in instance_configs:
                    future = executor.submit(
                        self.train_single_instance_advanced,
                        config, X_train, y_train, X_val, y_val
                    )
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Training"):
                    result = future.result()
                    results.append(result)
        
        # Analyze results
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            raise RuntimeError("No instances trained successfully!")
        
        # Find best model
        best_result = min(successful_results, key=lambda x: x['val_logloss'])
        
        # Calculate ensemble metrics
        ensemble_metrics = self.calculate_ensemble_metrics(successful_results, X_val, y_val)
        
        ensemble_results = {
            'n_instances': len(successful_results),
            'best_single': best_result,
            'all_results': successful_results,
            'ensemble_metrics': ensemble_metrics,
            'training_history': self.training_history,
            'total_training_time': sum(r['training_time'] for r in successful_results)
        }
        
        return ensemble_results
    
    def calculate_ensemble_metrics(self, results: List[Dict], X_val, y_val) -> Dict:
        """Calculate comprehensive ensemble performance metrics."""
        
        from sklearn.metrics import log_loss, accuracy_score
        import xgboost as xgb
        
        # Load all models and make predictions
        predictions = []
        models = []
        
        for result in results:
            if result['status'] == 'success':
                model = xgb.Booster()
                model.load_model(result['model_path'])
                models.append(model)
                
                dval = xgb.DMatrix(X_val, enable_categorical=False)
                pred = model.predict(dval)
                predictions.append(pred)
        
        if not predictions:
            return {'error': 'No successful predictions'}
        
        predictions = np.array(predictions)
        
        # Simple average ensemble
        avg_pred = np.mean(predictions, axis=0)
        avg_logloss = log_loss(y_val, avg_pred)
        avg_accuracy = accuracy_score(y_val, np.argmax(avg_pred, axis=1))
        
        # Weighted ensemble (by validation performance)
        weights = [1.0 / result['val_logloss'] for result in results if result['status'] == 'success']
        weights = np.array(weights) / sum(weights)
        
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        weighted_logloss = log_loss(y_val, weighted_pred)
        weighted_accuracy = accuracy_score(y_val, np.argmax(weighted_pred, axis=1))
        
        # Best single model performance
        best_single_logloss = min(r['val_logloss'] for r in results if r['status'] == 'success')
        
        return {
            'average_ensemble': {
                'logloss': avg_logloss,
                'accuracy': avg_accuracy
            },
            'weighted_ensemble': {
                'logloss': weighted_logloss,
                'accuracy': weighted_accuracy
            },
            'best_single': {
                'logloss': best_single_logloss
            },
            'improvement_over_best': {
                'average': best_single_logloss - avg_logloss,
                'weighted': best_single_logloss - weighted_logloss
            },
            'n_models': len(predictions)
        }
    
    def save_results(self, results: Dict, suffix: str = "") -> Path:
        """Save comprehensive training results."""
        timestamp = int(time.time())
        filename = f"multi_instance_results_{timestamp}{suffix}.json"
        filepath = RESULTS_DIR / filename
        
        # Prepare results for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"[Advanced Trainer] Results saved to: {filepath.relative_to(PROJECT_ROOT)}")
        return filepath


def main():
    """Main training function with comprehensive options."""
    parser = argparse.ArgumentParser(description="Advanced Multi-Instance XGBoost Training")
    parser.add_argument("--instances", type=int, help="Number of instances to train")
    parser.add_argument("--monitor", action="store_true", help="Enable GPU monitoring")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--ensemble-only", action="store_true", help="Skip single model training")
    parser.add_argument("--save-all", action="store_true", help="Save all intermediate results")
    
    args = parser.parse_args()
    
    # Run benchmark if requested
    if args.benchmark:
        print("="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        benchmark_results = benchmark_gpu_performance(n_samples=50000, n_features=200)
        
        benchmark_path = RESULTS_DIR / f"benchmark_{int(time.time())}.json"
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print(f"Benchmark results saved to: {benchmark_path.relative_to(PROJECT_ROOT)}")
        
        if not args.ensemble_only:
            return
    
    # Initialize trainer
    trainer = AdvancedMultiInstanceTrainer(
        n_instances=args.instances,
        enable_monitoring=args.monitor
    )
    
    # Load data
    X, y = trainer.load_and_prepare_data()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print("="*60)
    print("MULTI-INSTANCE ENSEMBLE TRAINING")
    print("="*60)
    
    # Train ensemble
    start_time = time.time()
    results = trainer.train_ensemble_advanced(X_train, y_train, X_val, y_val)
    total_time = time.time() - start_time
    
    # Add timing information
    results['meta'] = {
        'total_wall_time': total_time,
        'training_efficiency': results['total_training_time'] / total_time,
        'gpu_available': trainer.gpu_manager.gpu_available,
        'n_instances_requested': trainer.n_instances,
        'args': vars(args)
    }
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    print(f"Instances Trained: {results['n_instances']}")
    print(f"Best Single Model: {results['best_single']['val_logloss']:.5f}")
    
    if 'ensemble_metrics' in results:
        ensemble = results['ensemble_metrics']
        print(f"Average Ensemble: {ensemble['average_ensemble']['logloss']:.5f}")
        print(f"Weighted Ensemble: {ensemble['weighted_ensemble']['logloss']:.5f}")
        print(f"Best Improvement: {max(ensemble['improvement_over_best'].values()):.5f}")
    
    print(f"Total Training Time: {results['total_training_time']:.1f}s")
    print(f"Wall Clock Time: {total_time:.1f}s")
    print(f"Training Efficiency: {results['meta']['training_efficiency']:.1%}")
    
    # Save results
    results_path = trainer.save_results(results, "_final")
    
    # Copy best model to standard location
    import shutil
    best_model_path = results['best_single']['model_path']
    standard_path = MODEL_DIR / "football_predictor.json"
    shutil.copy(best_model_path, standard_path)
    
    print(f"\nBest model copied to: {standard_path.relative_to(PROJECT_ROOT)}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 