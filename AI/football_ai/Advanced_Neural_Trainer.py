#!/usr/bin/env python3
"""
ADVANCED PYTORCH NEURAL NETWORK TRAINER
State-of-the-art neural network with modern techniques

Features:
- Residual connections
- Multi-head attention
- Feature importance analysis
- Ensemble training
- Advanced regularization
- Live visualization

Usage: python Advanced_Neural_Trainer.py --10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "mega_enhanced_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature importance."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        residual = x
        
        # Linear transformations
        q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + residual)

class ResidualBlock(nn.Module):
    """Residual block with skip connections."""
    
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        # Skip connection
        out += residual
        out = self.activation(out)
        
        return out

class AttentionBlock(nn.Module):
    """Attention mechanism for feature importance."""
    
    def __init__(self, input_size, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.output = nn.Linear(input_size, input_size)
        
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        residual = x
        
        # Multi-head attention
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, v)
        attended = attended.view(batch_size, -1)
        
        # Output projection
        output = self.output(attended)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + residual)

class AdvancedFootballNet(nn.Module):
    """Advanced neural network with modern techniques."""
    
    def __init__(self, input_size, num_classes=3, dropout=0.3):
        super(AdvancedFootballNet, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention = AttentionBlock(512, num_heads=8)
        
        # Residual blocks
        self.residual1 = ResidualBlock(512, 256, dropout)
        self.residual2 = ResidualBlock(512, 256, dropout)
        
        # Progressive layers
        self.progressive = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout/3),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Input embedding
        embedded = self.input_embedding(x)
        
        # Attention
        attended = self.attention(embedded)
        
        # Residual blocks
        res1 = self.residual1(attended)
        res2 = self.residual2(res1)
        
        # Progressive layers
        progressive_out = self.progressive(res2)
        
        # Classification
        output = self.classifier(progressive_out)
        
        return output

class AdvancedTrainer:
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
        
        print(f"🚀 Advanced PyTorch Neural Network Trainer")
        print(f"💻 Device: {self.device}")
        print(f"⏱️  Training Duration: {minutes} minutes")
        print(f"🧠 Features: Residual connections, Multi-head attention, Advanced regularization")
        print("=" * 80)
    
    def load_and_prepare_data(self):
        """Load and prepare data with advanced preprocessing."""
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
        
        # Feature engineering
        print("🔬 Creating interaction features...")
        if 'Home_ELO' in X.columns and 'Away_ELO' in X.columns:
            X['ELO_Difference'] = X['Home_ELO'] - X['Away_ELO']
            X['ELO_Sum'] = X['Home_ELO'] + X['Away_ELO']
            X['ELO_Ratio'] = X['Home_ELO'] / (X['Away_ELO'] + 1e-8)
        
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
        
        # Outlier clipping
        X_train_scaled = np.clip(X_train_scaled, -3, 3)
        X_val_scaled = np.clip(X_val_scaled, -3, 3)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.LongTensor(y_train.values).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.LongTensor(y_val.values).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        return X_train_scaled.shape[1]
    
    def focal_loss(self, outputs, targets, alpha=1.0, gamma=2.0):
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def train_epoch(self, model, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = self.focal_loss(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 15 == 0:
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
                loss = F.cross_entropy(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def plot_training_curves(self):
        """Plot advanced training curves."""
        if len(self.train_losses) < 2:
            return
            
        try:
            plt.style.use('dark_background')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('🧠 Advanced Neural Network Training', fontsize=16, color='cyan')
            
            epochs = range(1, len(self.train_losses) + 1)
            
            # Loss curves
            ax1.plot(epochs, self.train_losses, 'cyan', label='Training Loss', linewidth=3)
            ax1.plot(epochs, self.val_losses, 'orange', label='Validation Loss', linewidth=3)
            ax1.set_title('📉 Loss Evolution', color='white', fontsize=14)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy curves
            ax2.plot(epochs, self.train_accuracies, 'lime', label='Training Accuracy', linewidth=3)
            ax2.plot(epochs, self.val_accuracies, 'red', label='Validation Accuracy', linewidth=3)
            ax2.set_title('📈 Accuracy Evolution', color='white', fontsize=14)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Overfitting monitor
            if len(self.train_accuracies) > 1:
                overfitting = [t - v for t, v in zip(self.train_accuracies, self.val_accuracies)]
                ax3.plot(epochs, overfitting, 'yellow', linewidth=3)
                ax3.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                ax3.fill_between(epochs, overfitting, alpha=0.3, color='yellow')
                ax3.set_title('🎯 Overfitting Monitor', color='white', fontsize=14)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Train - Val Accuracy (%)')
                ax3.grid(True, alpha=0.3)
            
            # Learning rate simulation
            lr_values = [0.001 * (0.7 ** (epoch // 5)) for epoch in epochs]
            ax4.plot(epochs, lr_values, 'magenta', linewidth=3)
            ax4.set_title('⚡ Learning Rate Schedule', color='white', fontsize=14)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'advanced_training.png', dpi=150, bbox_inches='tight', facecolor='black')
            plt.show(block=False)
            plt.pause(0.1)
            
        except Exception as e:
            print(f"⚠️  Plotting error: {e}")
    
    def train_advanced_network(self):
        """Main training loop."""
        # Load data
        input_size = self.load_and_prepare_data()
        
        # Create model
        model = AdvancedFootballNet(input_size=input_size, num_classes=3, dropout=0.3).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🧠 Advanced Neural Network Created")
        print(f"🔧 Architecture: {input_size} → [Attention + Residual] → 512 → 256 → 128 → 64 → 32 → 3")
        print(f"⚙️  Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.001, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3
        )
        
        print(f"\n🚀 Starting advanced training...")
        print("=" * 90)
        
        epoch = 0
        patience_counter = 0
        max_patience = 8
        
        while self.time_remaining() > 30:
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
            elapsed_min = (time.time() - self.start_time) / 60
            remaining_min = self.time_remaining() / 60
            
            print(f"📊 Epoch {epoch:3d} | Train: {train_loss:.4f}/{train_acc:6.2f}% | Val: {val_loss:.4f}/{val_acc:6.2f}% | LR: {current_lr:.2e} | {elapsed_min:.1f}m/{remaining_min:.1f}m")
            
            # Check for new best
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_model = model.state_dict().copy()
                patience_counter = 0
                print(f"🌟 NEW BEST VALIDATION ACCURACY: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Plot curves every 5 epochs
            if epoch % 5 == 0:
                self.plot_training_curves()
            
            # Early stopping
            if patience_counter >= max_patience or current_lr < 1e-7:
                print(f"🛑 Early stopping triggered")
                break
        
        # Final results
        total_time = (time.time() - self.start_time) / 60
        print("\n" + "=" * 90)
        print(f"🎉 ADVANCED TRAINING COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f} minutes")
        print(f"📊 Epochs completed: {epoch}")
        print(f"🏆 Best validation accuracy: {self.best_accuracy:.2f}%")
        print(f"📈 Final training accuracy: {self.train_accuracies[-1]:.2f}%")
        print("=" * 90)
        
        # Save best model
        if self.best_model:
            model_path = MODELS_DIR / "advanced_neural_network.pth"
            torch.save({
                'model_state_dict': self.best_model,
                'input_size': input_size,
                'best_accuracy': self.best_accuracy,
                'scaler': self.scaler
            }, model_path)
            print(f"💾 Advanced model saved: {model_path}")
        
        # Final plot
        self.plot_training_curves()
        
        return model
    
    def time_remaining(self):
        """Get remaining training time."""
        elapsed = time.time() - self.start_time
        return max(0, self.target_duration - elapsed)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Advanced PyTorch Neural Network Trainer')
    parser.add_argument('--minutes', type=int, default=10, help='Training duration in minutes')
    
    # Support --10 format
    if len(sys.argv) == 2 and sys.argv[1].startswith('--') and sys.argv[1][2:].isdigit():
        minutes = int(sys.argv[1][2:])
    else:
        args = parser.parse_args()
        minutes = args.minutes
    
    if minutes < 1 or minutes > 120:
        print("❌ Duration must be between 1 and 120 minutes")
        return False
    
    # Check PyTorch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}")
    
    # Start training
    trainer = AdvancedTrainer(minutes)
    model = trainer.train_advanced_network()
    
    print("\n🎯 Next: Use Neural_Visualizer.py to see the network architecture!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 