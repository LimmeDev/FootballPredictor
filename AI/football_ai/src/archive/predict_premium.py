#!/usr/bin/env python3
"""
Premium Football Match Predictor
Uses only pre-match features for realistic predictions
No data leakage from match outcomes
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "advanced_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

class PremiumFootballPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        self.metadata = {}
        
    def prepare_realistic_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features using only pre-match information."""
        print("🔧 Preparing realistic pre-match features...")
        
        # EXCLUDE match outcome features (data leakage)
        exclude_features = [
            'Home_Goals', 'Away_Goals', 'Total_Goals', 'Goal_Difference',
            'Home_Win_Margin', 'Away_Win_Margin', 'Home_HT_Goals', 'Away_HT_Goals',
            'Result', 'Result_Encoded', 'Date', 'HomeTeam', 'AwayTeam', 
            'League', 'Country', 'Season', 'Round', 'Source', 'MatchID'
        ]
        
        # INCLUDE only pre-match features
        include_features = [
            # Team Strength
            'Home_Market_Value', 'Away_Market_Value', 'Market_Value_Ratio',
            'Home_Avg_Age', 'Away_Avg_Age', 'Age_Difference',
            'Home_Tier', 'Away_Tier', 'Tier_Advantage',
            
            # UEFA Coefficients
            'Home_UEFA_Coeff', 'Away_UEFA_Coeff', 'UEFA_Coeff_Difference', 'UEFA_Quality_Match',
            
            # Recent Form
            'Home_Form_Points', 'Away_Form_Points', 'Form_Difference',
            'Home_Last_5_Points', 'Away_Last_5_Points', 'Recent_Form_Diff',
            
            # Injuries & Availability
            'Home_Injured_Count', 'Away_Injured_Count', 'Injury_Advantage',
            'Home_Suspended_Count', 'Away_Suspended_Count',
            'Home_Key_Players_Out', 'Away_Key_Players_Out', 'Availability_Score',
            
            # Head-to-Head History
            'H2H_Home_Wins', 'H2H_Draws', 'H2H_Away_Wins', 'H2H_Home_Win_Rate', 'H2H_Recent_Form',
            
            # Competition Context
            'Competition_Weight', 'Home_Advantage', 'Is_Derby',
            
            # Situational
            'Season_Stage', 'Is_European_Competition', 'Is_Weekend', 'Match_Importance',
        ]
        
        # Filter to only available columns
        available_features = [col for col in include_features if col in df.columns]
        print(f"📊 Using {len(available_features)} pre-match features")
        
        # Check for numeric columns only
        X = df[available_features].select_dtypes(include=[np.number])
        y = df['Result_Encoded']
        
        print(f"✅ Final feature set: {len(X.columns)} numeric features")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_realistic_model(self, test_size=0.2):
        """Train model using only pre-match features."""
        print("🚀 PREMIUM FOOTBALL PREDICTOR TRAINING")
        print("=" * 60)
        print("🎯 Using only PRE-MATCH features (no data leakage)")
        print("📊 Realistic prediction system")
        print("=" * 60)
        
        # Load data
        df = pd.read_parquet(DATA_FILE)
        print(f"📈 Loaded {len(df)} matches")
        
        # Prepare realistic features
        X, y = self.prepare_realistic_features(df)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"🔄 Training set: {len(X_train)} matches")
        print(f"🧪 Test set: {len(X_test)} matches")
        
        # Create realistic XGBoost model (anti-overfitting)
        self.model = xgb.XGBClassifier(
            device='cpu',
            tree_method='hist',
            n_jobs=8,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            
            # Anti-overfitting parameters
            max_depth=4,                    # Shallow trees
            learning_rate=0.05,             # Slow learning
            n_estimators=200,               # Moderate ensemble size
            subsample=0.8,                  # Row sampling
            colsample_bytree=0.8,           # Column sampling
            colsample_bylevel=0.8,          # Level sampling
            reg_alpha=1.0,                  # L1 regularization
            reg_lambda=2.0,                 # L2 regularization
            min_child_weight=5,             # Minimum samples per leaf
            gamma=0.1,                      # Minimum split loss
            eval_metric='mlogloss'
        )
        
        # Train without early stopping for CV compatibility
        self.model.fit(X_train, y_train, verbose=False)
        
        self.feature_columns = list(X.columns)
        self.is_trained = True
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"📊 Training accuracy: {train_score:.3f} ({train_score*100:.1f}%)")
        print(f"🧪 Test accuracy: {test_score:.3f} ({test_score*100:.1f}%)")
        
        # Cross-validation for robust evaluation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"📈 5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance
        print("\n🔍 Top 10 Most Important Pre-Match Features:")
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.3f}")
        
        # Save model and metadata
        self.save_model()
        
        return test_score
    
    def predict_match(self, home_team: str, away_team: str, 
                     match_context: Optional[Dict] = None) -> Dict:
        """Predict a specific match using pre-match features."""
        if not self.is_trained:
            model_file = MODELS_DIR / "premium_football_model.pkl"
            if model_file.exists():
                self.load_model()
            else:
                raise ValueError("Model not trained! Run train_realistic_model() first.")
        
        # Create match features
        match_features = self.create_match_features(home_team, away_team, match_context)
        
        # Predict probabilities
        proba = self.model.predict_proba([match_features])[0]
        
        # Map to results
        result_mapping = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        predictions = {result_mapping[i]: prob for i, prob in enumerate(proba)}
        
        # Add confidence and recommendation
        max_prob = max(proba)
        predicted_result = result_mapping[np.argmax(proba)]
        
        confidence_level = "Very High" if max_prob > 0.6 else \
                          "High" if max_prob > 0.5 else \
                          "Medium" if max_prob > 0.4 else "Low"
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predictions': predictions,
            'most_likely': predicted_result,
            'confidence': confidence_level,
            'max_probability': max_prob,
            'features_used': len(self.feature_columns)
        }
    
    def create_match_features(self, home_team: str, away_team: str, 
                            context: Optional[Dict] = None) -> List[float]:
        """Create feature vector for a match."""
        
        # Load auxiliary data
        team_strength_file = PROJECT_ROOT / "data" / "raw" / "team_strength.json"
        uefa_coeff_file = PROJECT_ROOT / "data" / "raw" / "uefa_coefficients.json"
        form_file = PROJECT_ROOT / "data" / "raw" / "team_form.json"
        injury_file = PROJECT_ROOT / "data" / "raw" / "injury_data.json"
        
        team_strength = {}
        uefa_coeffs = {}
        form_data = {}
        injury_data = {}
        
        if team_strength_file.exists():
            with open(team_strength_file, 'r') as f:
                team_strength = json.load(f)
        
        if uefa_coeff_file.exists():
            with open(uefa_coeff_file, 'r') as f:
                uefa_coeffs = json.load(f)
                
        if form_file.exists():
            with open(form_file, 'r') as f:
                form_data = json.load(f)
                
        if injury_file.exists():
            with open(injury_file, 'r') as f:
                injury_data = json.load(f)
        
        # Create feature vector (must match training features order)
        features = []
        
        # Team Strength Features
        home_strength = team_strength.get(home_team, {'market_value': 400, 'avg_age': 26.0, 'tier': 2})
        away_strength = team_strength.get(away_team, {'market_value': 400, 'avg_age': 26.0, 'tier': 2})
        
        features.extend([
            home_strength['market_value'],                     # Home_Market_Value
            away_strength['market_value'],                     # Away_Market_Value
            home_strength['market_value'] / away_strength['market_value'],  # Market_Value_Ratio
            home_strength['avg_age'],                          # Home_Avg_Age
            away_strength['avg_age'],                          # Away_Avg_Age
            home_strength['avg_age'] - away_strength['avg_age'],  # Age_Difference
            home_strength['tier'],                             # Home_Tier
            away_strength['tier'],                             # Away_Tier
            away_strength['tier'] - home_strength['tier'],     # Tier_Advantage
        ])
        
        # UEFA Coefficients
        home_uefa = uefa_coeffs.get(home_team, 30.0)
        away_uefa = uefa_coeffs.get(away_team, 30.0)
        
        features.extend([
            home_uefa,                                         # Home_UEFA_Coeff
            away_uefa,                                         # Away_UEFA_Coeff
            home_uefa - away_uefa,                             # UEFA_Coeff_Difference
            (home_uefa + away_uefa) / 2,                       # UEFA_Quality_Match
        ])
        
        # Form Features
        home_form = form_data.get(home_team, {'points_per_game': 1.5, 'last_5': [1.5]*5})
        away_form = form_data.get(away_team, {'points_per_game': 1.5, 'last_5': [1.5]*5})
        
        features.extend([
            home_form['points_per_game'],                      # Home_Form_Points
            away_form['points_per_game'],                      # Away_Form_Points
            home_form['points_per_game'] - away_form['points_per_game'],  # Form_Difference
            sum(home_form['last_5']),                          # Home_Last_5_Points
            sum(away_form['last_5']),                          # Away_Last_5_Points
            sum(home_form['last_5']) - sum(away_form['last_5']), # Recent_Form_Diff
        ])
        
        # Injury Features
        home_injuries = injury_data.get(home_team, {'injured': 1, 'suspended': 0, 'key_players_out': []})
        away_injuries = injury_data.get(away_team, {'injured': 1, 'suspended': 0, 'key_players_out': []})
        
        features.extend([
            home_injuries['injured'],                          # Home_Injured_Count
            away_injuries['injured'],                          # Away_Injured_Count
            away_injuries['injured'] - home_injuries['injured'], # Injury_Advantage
            home_injuries['suspended'],                        # Home_Suspended_Count
            away_injuries['suspended'],                        # Away_Suspended_Count
            len(home_injuries['key_players_out']),             # Home_Key_Players_Out
            len(away_injuries['key_players_out']),             # Away_Key_Players_Out
            ((5 - home_injuries['injured'] - home_injuries['suspended'] - len(home_injuries['key_players_out'])) -
             (5 - away_injuries['injured'] - away_injuries['suspended'] - len(away_injuries['key_players_out']))), # Availability_Score
        ])
        
        # H2H Features (simplified)
        features.extend([
            1, 1, 1,        # H2H_Home_Wins, H2H_Draws, H2H_Away_Wins
            0.33,           # H2H_Home_Win_Rate
            0,              # H2H_Recent_Form
        ])
        
        # Competition Features
        competition_weight = context.get('competition_weight', 0.7) if context else 0.7
        is_derby = 1 if self.is_derby_match(home_team, away_team) else 0
        
        features.extend([
            competition_weight,                                # Competition_Weight
            1.4,                                              # Home_Advantage
            is_derby,                                         # Is_Derby
        ])
        
        # Situational Features
        features.extend([
            2,              # Season_Stage (mid-season)
            0,              # Is_European_Competition
            1,              # Is_Weekend
            1.0,            # Match_Importance
        ])
        
        return features
    
    def is_derby_match(self, home_team: str, away_team: str) -> bool:
        """Check if it's a derby match."""
        derby_pairs = [
            ('Manchester City', 'Manchester United'),
            ('Liverpool', 'Everton'),
            ('Arsenal', 'Tottenham'),
            ('Barcelona', 'Real Madrid'),
            ('Milan', 'Inter'),
            ('Bayern München', 'Borussia Dortmund'),
        ]
        
        for team1, team2 in derby_pairs:
            if (home_team == team1 and away_team == team2) or \
               (home_team == team2 and away_team == team1):
                return True
        return False
    
    def save_model(self):
        """Save the trained model and metadata."""
        model_file = MODELS_DIR / "premium_football_model.pkl"
        metadata_file = MODELS_DIR / "premium_model_metadata.json"
        
        # Save model
        joblib.dump(self.model, model_file)
        
        # Save metadata
        self.metadata = {
            'model_type': 'Premium XGBoost (Pre-match only)',
            'features': self.feature_columns,
            'feature_count': len(self.feature_columns),
            'training_date': pd.Timestamp.now().isoformat(),
            'anti_overfitting': True,
            'data_leakage_free': True
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Model saved to: {model_file}")
        print(f"📋 Metadata saved to: {metadata_file}")
    
    def load_model(self):
        """Load a previously trained model."""
        model_file = MODELS_DIR / "premium_football_model.pkl"
        metadata_file = MODELS_DIR / "premium_model_metadata.json"
        
        if model_file.exists() and metadata_file.exists():
            self.model = joblib.load(model_file)
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata['features']
            self.is_trained = True
            print(f"✅ Loaded model: {self.metadata['model_type']}")
        else:
            raise FileNotFoundError("No trained model found!")

def main():
    """Main function for training and demonstration."""
    predictor = PremiumFootballPredictor()
    
    # Train the model
    test_accuracy = predictor.train_realistic_model()
    
    print(f"\n🎉 Premium model training complete!")
    print(f"🏆 Test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    
    # Demo predictions
    print("\n🔮 DEMO PREDICTIONS")
    print("=" * 50)
    
    matches = [
        ("Manchester City", "Liverpool"),
        ("Arsenal", "Chelsea"),
        ("Barcelona", "Real Madrid"),
        ("Bayern München", "Borussia Dortmund"),
        ("Manchester United", "Tottenham"),
    ]
    
    for home, away in matches:
        try:
            prediction = predictor.predict_match(home, away)
            print(f"\n⚽ {home} vs {away}")
            print(f"   🏠 Home Win: {prediction['predictions']['Home Win']:.1%}")
            print(f"   🤝 Draw: {prediction['predictions']['Draw']:.1%}")
            print(f"   ✈️  Away Win: {prediction['predictions']['Away Win']:.1%}")
            print(f"   🎯 Most Likely: {prediction['most_likely']} ({prediction['confidence']} confidence)")
        except Exception as e:
            print(f"❌ Error predicting {home} vs {away}: {e}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 