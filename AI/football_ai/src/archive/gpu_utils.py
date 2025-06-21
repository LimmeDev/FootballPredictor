"""GPU utilities for XGBoost multi-instance training and memory management.

This module provides utilities for:
- GPU memory monitoring and allocation
- Multi-instance coordination on single GPU
- Performance optimization for XGBoost GPU training
- Resource management and cleanup
"""
from __future__ import annotations

import gc
import os
import time
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import xgboost as xgb


class GPUResourceManager:
    """Manages GPU resources for multi-instance XGBoost training."""
    
    def __init__(self, max_memory_fraction: float = 0.8):
        self.max_memory_fraction = max_memory_fraction
        self.active_instances = {}
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for XGBoost."""
        try:
            # Test GPU availability with a small DMatrix
            test_data = np.random.rand(100, 10)
            test_labels = np.random.randint(0, 3, 100)
            
            dtrain = xgb.DMatrix(test_data, label=test_labels)
            
            params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'objective': 'multi:softprob',
                'num_class': 3,
                'verbosity': 0
            }
            
            # Try to train a small model
            model = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
            del model, dtrain
            gc.collect()
            
            return True
        except Exception as e:
            print(f"[GPU ResourceManager] GPU not available: {e}")
            return False
    
    def get_optimal_instances(self, total_memory_gb: float = None) -> int:
        """Calculate optimal number of instances based on available GPU memory."""
        if not self.gpu_available:
            return 1
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            total_memory = meminfo.total / (1024**3)  # Convert to GB
            available_memory = meminfo.free / (1024**3)  # Convert to GB
            
            # Estimate memory per instance (rough heuristic)
            memory_per_instance = min(2.0, available_memory * 0.25)  # Max 2GB per instance
            max_instances = int(available_memory * self.max_memory_fraction / memory_per_instance)
            
            # Cap at reasonable limits
            return min(max_instances, 6)
            
        except ImportError:
            print("[GPU ResourceManager] pynvml not available, using default instances")
            return 4
        except Exception as e:
            print(f"[GPU ResourceManager] Error checking GPU memory: {e}")
            return 2
    
    @contextmanager
    def gpu_context(self, instance_id: int):
        """Context manager for GPU instance management."""
        if not self.gpu_available:
            yield
            return
        
        try:
            # Set up GPU environment for this instance
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # Register instance
            self.active_instances[instance_id] = {
                'start_time': time.time(),
                'pid': os.getpid()
            }
            
            yield
            
        finally:
            # Cleanup
            if instance_id in self.active_instances:
                del self.active_instances[instance_id]
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to avoid GPU memory conflicts
            time.sleep(0.1)
    
    def get_optimized_params(self, base_params: Dict, instance_id: int = 0) -> Dict:
        """Get GPU-optimized parameters for XGBoost."""
        optimized = base_params.copy()
        
        if self.gpu_available:
            optimized.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor',
                'single_precision_histogram': True,
                'max_bin': 256,  # Optimal for GPU
                'n_jobs': 1,  # Important for multi-instance
            })
        else:
            # Fallback to CPU optimized parameters
            optimized.update({
                'tree_method': 'hist',
                'n_jobs': min(4, psutil.cpu_count()),
            })
            
            # Remove GPU-specific parameters
            gpu_params = ['gpu_id', 'predictor', 'single_precision_histogram']
            for param in gpu_params:
                optimized.pop(param, None)
        
        return optimized
    
    def monitor_gpu_usage(self) -> Dict:
        """Monitor current GPU usage."""
        if not self.gpu_available:
            return {'gpu_available': False}
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            return {
                'gpu_available': True,
                'memory_total': meminfo.total,
                'memory_used': meminfo.used,
                'memory_free': meminfo.free,
                'memory_utilization_percent': (meminfo.used / meminfo.total) * 100,
                'gpu_utilization_percent': utilization.gpu,
                'active_instances': len(self.active_instances),
                'instance_details': self.active_instances.copy()
            }
        except ImportError:
            return {
                'gpu_available': True,
                'pynvml_available': False,
                'active_instances': len(self.active_instances)
            }
        except Exception as e:
            return {
                'gpu_available': True,
                'monitoring_error': str(e),
                'active_instances': len(self.active_instances)
            }


class XGBoostGPUTrainer:
    """XGBoost trainer optimized for GPU multi-instance training."""
    
    def __init__(self, gpu_manager: GPUResourceManager = None):
        self.gpu_manager = gpu_manager or GPUResourceManager()
        
    def create_optimized_dmatrix(self, X, y=None, **kwargs) -> xgb.DMatrix:
        """Create GPU-optimized DMatrix."""
        # Ensure data is in the right format
        if hasattr(X, 'values'):
            X = X.values
        if y is not None and hasattr(y, 'values'):
            y = y.values
        
        # Create DMatrix with GPU optimizations
        dmatrix = xgb.DMatrix(
            X, 
            label=y, 
            enable_categorical=False,
            **kwargs
        )
        
        return dmatrix
    
    def train_with_gpu_optimization(self, 
                                  params: Dict, 
                                  dtrain: xgb.DMatrix,
                                  num_boost_round: int = 1000,
                                  evals: List[Tuple] = None,
                                  instance_id: int = 0,
                                  **kwargs) -> xgb.Booster:
        """Train XGBoost model with GPU optimization."""
        
        with self.gpu_manager.gpu_context(instance_id):
            # Get optimized parameters
            optimized_params = self.gpu_manager.get_optimized_params(params, instance_id)
            
            # Train model
            model = xgb.train(
                optimized_params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evals or [(dtrain, 'train')],
                verbose_eval=False,
                **kwargs
            )
            
            return model
    
    def cross_validate_gpu(self, 
                          params: Dict,
                          dtrain: xgb.DMatrix,
                          num_boost_round: int = 1000,
                          nfold: int = 5,
                          **kwargs) -> Dict:
        """Perform cross-validation with GPU optimization."""
        
        # Get optimized parameters
        optimized_params = self.gpu_manager.get_optimized_params(params)
        
        # Perform CV
        cv_results = xgb.cv(
            optimized_params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=nfold,
            shuffle=True,
            seed=42,
            verbose_eval=False,
            **kwargs
        )
        
        return cv_results


def benchmark_gpu_performance(n_samples: int = 10000, n_features: int = 100) -> Dict:
    """Benchmark GPU vs CPU performance for XGBoost."""
    print(f"[Benchmark] Testing with {n_samples} samples, {n_features} features")
    
    # Generate test data
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    results = {}
    
    # Test GPU
    gpu_manager = GPUResourceManager()
    trainer = XGBoostGPUTrainer(gpu_manager)
    
    if gpu_manager.gpu_available:
        print("[Benchmark] Testing GPU performance...")
        dtrain = trainer.create_optimized_dmatrix(X, y)
        
        gpu_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
        }
        
        start_time = time.time()
        gpu_model = trainer.train_with_gpu_optimization(
            gpu_params, dtrain, num_boost_round=100
        )
        gpu_time = time.time() - start_time
        
        results['gpu'] = {
            'available': True,
            'training_time': gpu_time,
            'samples_per_second': n_samples / gpu_time
        }
    else:
        results['gpu'] = {'available': False}
    
    # Test CPU
    print("[Benchmark] Testing CPU performance...")
    cpu_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.1,
        'tree_method': 'hist',
        'n_jobs': psutil.cpu_count()
    }
    
    dtrain_cpu = xgb.DMatrix(X, label=y)
    
    start_time = time.time()
    cpu_model = xgb.train(cpu_params, dtrain_cpu, num_boost_round=100, verbose_eval=False)
    cpu_time = time.time() - start_time
    
    results['cpu'] = {
        'training_time': cpu_time,
        'samples_per_second': n_samples / cpu_time
    }
    
    # Calculate speedup
    if results['gpu']['available']:
        speedup = cpu_time / results['gpu']['training_time']
        results['speedup'] = {
            'gpu_vs_cpu': speedup,
            'performance_gain': f"{speedup:.2f}x faster" if speedup > 1 else f"{1/speedup:.2f}x slower"
        }
    
    return results


if __name__ == "__main__":
    # Run benchmarks and tests
    print("="*60)
    print("XGBoost GPU Multi-Instance Utilities")
    print("="*60)
    
    # Initialize GPU manager
    gpu_manager = GPUResourceManager()
    
    print(f"GPU Available: {gpu_manager.gpu_available}")
    print(f"Optimal Instances: {gpu_manager.get_optimal_instances()}")
    
    # Monitor GPU usage
    gpu_status = gpu_manager.monitor_gpu_usage()
    print("\nGPU Status:")
    for key, value in gpu_status.items():
        print(f"  {key}: {value}")
    
    # Run benchmark
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    benchmark_results = benchmark_gpu_performance()
    
    print("\nBenchmark Results:")
    for device, stats in benchmark_results.items():
        if device in ['gpu', 'cpu']:
            print(f"\n{device.upper()}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    if 'speedup' in benchmark_results:
        print(f"\nPerformance: {benchmark_results['speedup']['performance_gain']}") 