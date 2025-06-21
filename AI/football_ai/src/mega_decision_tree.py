#!/usr/bin/env python3
"""
MEGA DECISION TREE SYSTEM
Build the largest possible decision tree for football predictions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "mega_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

class MegaDecisionTreeSystem:
    def __init__(self):
        self.models = {}
        self.feature_importances = {}
        
    def load_mega_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load mega features dataset."""
        if not DATA_FILE.exists():
            raise FileNotFoundError("Mega features not found! Run mega_features.py first.")
        
        df = pd.read_parquet(DATA_FILE)
        print(f"📊 Loaded {len(df)} matches with {len(df.columns)} columns")
        
        # Exclude non-feature columns for ML
        exclude_cols = [
            'Result', 'Result_Encoded', 'Date', 'HomeTeam', 'AwayTeam', 
            'League', 'Country', 'Season', 'Round', 'Source', 'MatchID',
            'Home_Formation', 'Away_Formation', 'Weather'  # Categorical
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Only numeric features
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df['Result_Encoded']
        
        print(f"✅ Using {len(X.columns)} numeric features")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def build_maximum_decision_tree(self, X: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
        """Build the largest possible decision tree."""
        print("🌳 Building MAXIMUM decision tree...")
        
        # Maximum complexity parameters
        model = DecisionTreeClassifier(
            criterion='gini',
            splitter='best',
            max_depth=None,                # No depth limit!
            min_samples_split=2,           # Minimum possible
            min_samples_leaf=1,            # Minimum possible
            min_weight_fraction_leaf=0.0,
            max_features=None,             # Use ALL features
            random_state=42,
            max_leaf_nodes=None,           # No limit
            min_impurity_decrease=0.0,     # No pruning
            class_weight=None,
            ccp_alpha=0.0                  # No cost complexity pruning
        )
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"🔄 Training on {len(X_train)} matches with {len(X.columns)} features")
        print(f"🧪 Testing on {len(X_test)} matches")
        
        # Fit the massive tree
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"📊 Training accuracy: {train_score:.3f} ({train_score*100:.1f}%)")
        print(f"🧪 Test accuracy: {test_score:.3f} ({test_score*100:.1f}%)")
        print(f"🌿 Tree depth: {model.tree_.max_depth}")
        print(f"🍃 Number of leaves: {model.tree_.n_leaves}")
        print(f"🌳 Number of nodes: {model.tree_.node_count}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 20 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(20).iterrows()):
            print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        self.models['mega_tree'] = model
        self.feature_importances['mega_tree'] = feature_importance
        
        return model
    
    def build_random_forest_ensemble(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """Build massive random forest ensemble."""
        print("\n🌲 Building MEGA Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=500,              # 500 trees (reduced for VM)
            criterion='gini',
            max_depth=None,                # No depth limit
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',           # sqrt(features) per split
            bootstrap=True,
            oob_score=True,
            n_jobs=8,                      # Use 8 cores
            random_state=42,
            verbose=1
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"🔄 Training 500 trees on {len(X_train)} matches...")
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        oob_score = model.oob_score_
        
        print(f"📊 Training accuracy: {train_score:.3f} ({train_score*100:.1f}%)")
        print(f"🧪 Test accuracy: {test_score:.3f} ({test_score*100:.1f}%)")
        print(f"🎯 OOB accuracy: {oob_score:.3f} ({oob_score*100:.1f}%)")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 15 Forest Features:")
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        self.models['mega_forest'] = model
        self.feature_importances['mega_forest'] = feature_importance
        
        return model
    
    def extract_decision_rules(self, model: DecisionTreeClassifier, X: pd.DataFrame):
        """Extract readable decision rules from tree."""
        print(f"\n📋 Extracting decision rules...")
        
        tree_rules = export_text(model, feature_names=list(X.columns), max_depth=3)
        
        rules_file = MODELS_DIR / "mega_tree_rules.txt"
        with open(rules_file, 'w') as f:
            f.write(tree_rules)
        
        print(f"💾 Saved decision rules to: {rules_file}")
        
        # Show sample rules
        rules_lines = tree_rules.split('\n')[:20]
        print(f"\n📜 Sample Decision Rules:")
        for line in rules_lines[:10]:
            if line.strip():
                print(f"   {line}")
    
    def save_models(self):
        """Save all trained models."""
        print("\n💾 Saving mega models...")
        
        for name, model in self.models.items():
            model_file = MODELS_DIR / f"{name}_model.pkl"
            joblib.dump(model, model_file)
            print(f"✅ Saved {name} to: {model_file}")
        
        # Save feature importances
        importance_file = MODELS_DIR / "mega_feature_importances.json"
        importance_dict = {}
        for name, importance_df in self.feature_importances.items():
            importance_dict[name] = importance_df.to_dict('records')
        
        with open(importance_file, 'w') as f:
            json.dump(importance_dict, f, indent=2)
        
        print(f"📊 Saved feature importances to: {importance_file}")

def main():
    """Main function for mega decision tree system."""
    print("🌳 MEGA DECISION TREE SYSTEM")
    print("=" * 80)
    print("🎯 Building the largest possible decision trees")
    print("📊 Using 200+ features for maximum complexity")
    print("🌲 Maximum tree depth and ensemble methods")
    print("=" * 80)
    
    # Initialize system
    mega_system = MegaDecisionTreeSystem()
    
    # Load mega data
    X, y = mega_system.load_mega_data()
    
    # Build all mega models
    print(f"\n🚀 Building mega models with {len(X.columns)} features...")
    
    # 1. Maximum single decision tree
    mega_tree = mega_system.build_maximum_decision_tree(X, y)
    
    # 2. Random Forest ensemble
    mega_forest = mega_system.build_random_forest_ensemble(X, y)
    
    # Extract decision rules
    mega_system.extract_decision_rules(mega_tree, X)
    
    # Save all models
    mega_system.save_models()
    
    print(f"\n🎉 MEGA DECISION TREE SYSTEM COMPLETE!")
    print(f"🌳 Built 2 mega models with {len(X.columns)} features")
    print(f"📊 Maximum tree complexity achieved!")
    print(f"🎯 Ready for ultimate football predictions!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 