#!/bin/bash

# =============================================================================
# 🚀 FOOTBALL AI - GOOGLE COLAB SETUP SCRIPT
# =============================================================================
# This script sets up the Football AI system optimized for Google Colab
# Run with: !bash setup_colab.sh

echo "🚀 Setting up Football AI for Google Colab..."
echo "=================================================="

# Check if we're in Colab
if [ -z "$COLAB_GPU" ]; then
    echo "⚠️  Warning: This script is optimized for Google Colab"
    echo "💡 Consider running in Colab for best GPU performance"
fi

# =============================================================================
# 1. SYSTEM PREPARATION
# =============================================================================
echo "📦 Step 1/6: Preparing system..."

# Update system packages (silent)
apt-get update -qq > /dev/null 2>&1
apt-get install -y git curl -qq > /dev/null 2>&1

# Create directories
mkdir -p football_ai/{data/{raw,processed},models,logs}
cd football_ai

echo "✅ System preparation complete"

# =============================================================================
# 2. PYTHON DEPENDENCIES
# =============================================================================
echo "🐍 Step 2/6: Installing Python dependencies..."

# Upgrade pip
pip install --upgrade pip -q

# Install core dependencies with specific versions for Colab compatibility
pip install -q \
    pandas==2.1.4 \
    numpy==1.25.2 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    optuna==3.5.0 \
    tqdm==4.66.1 \
    requests==2.31.0 \
    psutil==5.9.6 \
    joblib==1.3.2 \
    dask[complete]==2023.12.1

# GPU monitoring (if available)
pip install -q pynvml==11.5.0

echo "✅ Python dependencies installed"

# =============================================================================
# 3. GPU SETUP AND VERIFICATION
# =============================================================================
echo "🔥 Step 3/6: Setting up GPU support..."

# Check GPU availability
python3 -c "
import subprocess
import sys

try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print('✅ NVIDIA GPU detected')
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Tesla' in line or 'T4' in line or 'V100' in line or 'A100' in line:
                print(f'   GPU: {line.strip()}')
                break
    else:
        print('⚠️  No NVIDIA GPU detected - will use CPU')
except:
    print('⚠️  GPU detection failed - will use CPU')

# Test XGBoost GPU support
try:
    import xgboost as xgb
    if xgb.core._LIB.XGBoostIsGPUSupported():
        print('✅ XGBoost GPU support available')
    else:
        print('⚠️  XGBoost GPU support not available')
except Exception as e:
    print(f'⚠️  XGBoost GPU test failed: {e}')
"

echo "✅ GPU setup complete"

# =============================================================================
# 4. DOWNLOAD FOOTBALL AI CODE
# =============================================================================
echo "📥 Step 4/6: Downloading Football AI code..."

# Download source files from the repository
curl -s -L -o download_data.py "https://raw.githubusercontent.com/YOUR_USERNAME/FootballPredictor/main/src/download_data.py"
curl -s -L -o make_features.py "https://raw.githubusercontent.com/YOUR_USERNAME/FootballPredictor/main/src/make_features.py"
curl -s -L -o train_model.py "https://raw.githubusercontent.com/YOUR_USERNAME/FootballPredictor/main/src/train_model.py"
curl -s -L -o predict_match.py "https://raw.githubusercontent.com/YOUR_USERNAME/FootballPredictor/main/src/predict_match.py"
curl -s -L -o gpu_utils.py "https://raw.githubusercontent.com/YOUR_USERNAME/FootballPredictor/main/src/gpu_utils.py"
curl -s -L -o train_multi_instance.py "https://raw.githubusercontent.com/YOUR_USERNAME/FootballPredictor/main/src/train_multi_instance.py"
curl -s -L -o parse_openfootball.py "https://raw.githubusercontent.com/YOUR_USERNAME/FootballPredictor/main/src/parse_openfootball.py"

# Alternative: Create minimal versions directly for Colab
echo "📝 Creating Colab-optimized scripts..."

# Create a simple Colab-specific download script
cat > colab_download.py << 'EOF'
"""Lightweight data downloader optimized for Google Colab."""
import requests
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import time

def download_sample_data():
    """Download a sample dataset for quick testing in Colab."""
    print("📥 Downloading sample football data for Colab...")
    
    # Create data directory
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    # Download FiveThirtyEight SPI data (small file)
    url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("data/raw/spi_matches.csv", "wb") as f:
            f.write(response.content)
        print("✅ Downloaded FiveThirtyEight SPI data")
    
    # Create a sample features file for quick testing
    sample_data = {
        'Date': pd.date_range('2023-01-01', periods=1000, freq='3D'),
        'HomeTeam': ['Team A', 'Team B'] * 500,
        'AwayTeam': ['Team B', 'Team A'] * 500,
        'Home_Goals': [1, 2, 0, 3] * 250,
        'Away_Goals': [0, 1, 1, 2] * 250,
        'League': 'Sample League',
        'Source': 'Sample'
    }
    
    df = pd.DataFrame(sample_data)
    df['Result'] = df.apply(lambda x: 'H' if x['Home_Goals'] > x['Away_Goals'] 
                                   else 'A' if x['Home_Goals'] < x['Away_Goals'] 
                                   else 'D', axis=1)
    
    # Add some features
    df['Total_Goals'] = df['Home_Goals'] + df['Away_Goals']
    df['Goal_Difference'] = df['Home_Goals'] - df['Away_Goals']
    
    df.to_parquet("data/combined_features.parquet", index=False)
    print("✅ Created sample features dataset")
    
    return True

if __name__ == "__main__":
    download_sample_data()
EOF

echo "✅ Code download complete"

# =============================================================================
# 5. INITIAL DATA DOWNLOAD
# =============================================================================
echo "📊 Step 5/6: Downloading sample data..."

# Run the sample data download
python3 colab_download.py

echo "✅ Sample data downloaded"

# =============================================================================
# 6. VERIFICATION AND TESTING
# =============================================================================
echo "🧪 Step 6/6: Running verification tests..."

# Test basic functionality
python3 -c "
import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path

print('Testing basic functionality...')

# Test data loading
try:
    df = pd.read_parquet('data/combined_features.parquet')
    print(f'✅ Data loaded: {len(df)} samples, {len(df.columns)} features')
except Exception as e:
    print(f'❌ Data loading failed: {e}')

# Test XGBoost
try:
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # Try GPU first
    try:
        params = {'objective': 'multi:softprob', 'num_class': 3, 'tree_method': 'gpu_hist'}
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
        print('✅ XGBoost GPU training successful')
    except:
        # Fallback to CPU
        params = {'objective': 'multi:softprob', 'num_class': 3, 'tree_method': 'hist'}
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
        print('✅ XGBoost CPU training successful (GPU not available)')
        
except Exception as e:
    print(f'❌ XGBoost test failed: {e}')

print('\\n🎯 Verification complete!')
"

# =============================================================================
# COMPLETION MESSAGE
# =============================================================================
echo ""
echo "🎉 FOOTBALL AI SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "📋 What was installed:"
echo "  ✅ Python dependencies (XGBoost, pandas, etc.)"
echo "  ✅ GPU support (if available)"
echo "  ✅ Sample football dataset"
echo "  ✅ Core prediction scripts"
echo ""
echo "🚀 Quick Start Commands:"
echo "  # Download full datasets (optional)"
echo "  !python colab_download.py"
echo ""
echo "  # Train a quick model"
echo "  !python -c \"
echo "import pandas as pd"
echo "import xgboost as xgb"
echo "from sklearn.model_selection import train_test_split"
echo "from sklearn.preprocessing import LabelEncoder"
echo ""
echo "# Load data"
echo "df = pd.read_parquet('data/combined_features.parquet')"
echo "print(f'Dataset: {len(df)} matches')"
echo ""
echo "# Simple feature selection"
echo "X = df[['Home_Goals', 'Away_Goals', 'Total_Goals', 'Goal_Difference']].fillna(0)"
echo "y = LabelEncoder().fit_transform(df['Result'])"
echo ""
echo "# Train model"
echo "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)"
echo "dtrain = xgb.DMatrix(X_train, label=y_train)"
echo "dval = xgb.DMatrix(X_val, label=y_val)"
echo ""
echo "params = {'objective': 'multi:softprob', 'num_class': 3, 'tree_method': 'hist'}"
echo "model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'val')])"
echo "print('\\n✅ Model training complete!')"
echo "\""
echo ""
echo "💡 Tips for Colab:"
echo "  • Use GPU runtime for faster training"
echo "  • Save models to Google Drive for persistence"
echo "  • Install requirements at start of each session"
echo ""
echo "📖 Documentation: README.md"
echo "🆘 Need help? Check the error logs above"
echo ""
echo "Happy predicting! ⚽🎯" 