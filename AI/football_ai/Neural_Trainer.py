#!/usr/bin/env python3
"""
PYTORCH NEURAL NETWORK TRAINER
Live training with real-time progress and loss curves

Usage: python Neural_Trainer.py --20    (train for 20 minutes)
       python Neural_Trainer.py --5     (train for 5 minutes)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "mega_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

class FootballNet(nn.Module):
    """Deep Neural Network for Football Prediction"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], num_classes=3, dropout=0.3):
        super(FootballNet, self).__init__()
        
        # Build dynamic architecture
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class LiveTrainer:
    def __init__(self, minutes: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_duration = minutes * 60
        self.start_time = time.time()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_accuracy = 0
        self.best_model = None
        
        print(f"🚀 PyTorch Neural Network Trainer")
        print(f"💻 Device: {self.device}")
        print(f"⏱️  Training Duration: {minutes} minutes")
        print("=" * 60)
    
    def load_and_prepare_data(self):
        """Load and prepare data for neural network training."""
        print("📊 Loading mega features...")
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
        
        print(f"✅ Dataset: {len(df)} matches, {len(X.columns)} features")
        print(f"🎯 Classes: {dict(y.value_counts())}")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.LongTensor(y_train.values).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.LongTensor(y_val.values).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        return X_train_scaled.shape[1]  # Number of features
    
    def create_model(self, input_size):
        """Create and initialize the neural network."""
        print(f"🧠 Creating neural network...")
        
        # Dynamic architecture based on input size
        if input_size > 150:
            hidden_sizes = [512, 256, 128, 64, 32]
        elif input_size > 100:
            hidden_sizes = [256, 128, 64, 32]
        else:
            hidden_sizes = [128, 64, 32]
        
        model = FootballNet(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=3,
            dropout=0.3
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🔧 Architecture: {input_size} → {' → '.join(map(str, hidden_sizes))} → 3")
        print(f"⚙️  Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, model, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Show progress every 10 batches
            if batch_idx % 10 == 0:
                progress = (batch_idx / len(self.train_loader)) * 100
                print(f"\r🔥 Training: [{progress:6.1f}%] Loss: {loss.item():.4f} Acc: {100.*correct/total:6.2f}%", end='', flush=True)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        print(f"\r🔥 Training Complete: Loss: {avg_loss:.4f} Acc: {accuracy:6.2f}%")
        
        return avg_loss, accuracy
    
    def validate(self, model, criterion):
        """Validate the model."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def time_remaining(self):
        """Get remaining training time."""
        elapsed = time.time() - self.start_time
        return max(0, self.target_duration - elapsed)
    
    def print_live_stats(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Print live training statistics."""
        elapsed_min = (time.time() - self.start_time) / 60
        remaining_min = self.time_remaining() / 60
        
        print(f"📊 Epoch {epoch:3d} | Train: {train_loss:.4f}/{train_acc:6.2f}% | Val: {val_loss:.4f}/{val_acc:6.2f}% | LR: {lr:.2e} | {elapsed_min:.1f}m/{remaining_min:.1f}m")
        
        # Check for new best
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            print(f"🌟 NEW BEST VALIDATION ACCURACY: {val_acc:.2f}%")
    
    def plot_training_curves(self):
        """Plot live training curves."""
        if len(self.train_losses) < 2:
            return
            
        try:
            plt.clf()
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss curves
            epochs = range(1, len(self.train_losses) + 1)
            ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy curves
            ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
            ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'training_curves.png', dpi=100, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(0.1)
            
        except Exception as e:
            print(f"⚠️  Plotting error: {e}")
    
    def train_neural_network(self):
        """Main training loop."""
        # Load data
        input_size = self.load_and_prepare_data()
        
        # Create model
        model = self.create_model(input_size)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
        
        print(f"\n🚀 Starting training loop...")
        print("=" * 80)
        
        epoch = 0
        while self.time_remaining() > 30:  # Keep 30s buffer
            epoch += 1
            
            # Training
            train_loss, train_acc = self.train_epoch(model, optimizer, criterion)
            
            # Validation
            val_loss, val_acc = self.validate(model, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print stats
            self.print_live_stats(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_model = model.state_dict().copy()
            
            # Plot curves every 5 epochs
            if epoch % 5 == 0:
                self.plot_training_curves()
            
            # Early stopping check
            if current_lr < 1e-6:
                print("🛑 Learning rate too small, stopping...")
                break
        
        # Final results
        total_time = (time.time() - self.start_time) / 60
        print("\n" + "=" * 80)
        print(f"🎉 TRAINING COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f} minutes")
        print(f"📊 Epochs completed: {epoch}")
        print(f"🏆 Best validation accuracy: {self.best_accuracy:.2f}%")
        print(f"📈 Final training accuracy: {self.train_accuracies[-1]:.2f}%")
        print("=" * 80)
        
        # Save best model
        if self.best_model:
            model_path = MODELS_DIR / "best_neural_network.pth"
            torch.save({
                'model_state_dict': self.best_model,
                'input_size': input_size,
                'best_accuracy': self.best_accuracy,
                'scaler': self.scaler
            }, model_path)
            print(f"💾 Best model saved: {model_path}")
        
        # Final plot
        self.plot_training_curves()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='PyTorch Neural Network Trainer')
    parser.add_argument('--minutes', type=int, default=20, help='Training duration in minutes')
    
    # Support --20 format
    if len(sys.argv) == 2 and sys.argv[1].startswith('--') and sys.argv[1][2:].isdigit():
        minutes = int(sys.argv[1][2:])
    else:
        args = parser.parse_args()
        minutes = args.minutes
    
    if minutes < 1 or minutes > 1440:
        print("❌ Duration must be between 1 and 1440 minutes")
        return False
    
    # Check PyTorch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
    
    # Start training
    trainer = LiveTrainer(minutes)
    trainer.train_neural_network()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 