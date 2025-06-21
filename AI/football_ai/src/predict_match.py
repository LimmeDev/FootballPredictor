"""Predict outcome probabilities for a single fixture using XGBoost.

Usage
-----
1. Legacy – provide a JSON file that already contains a full feature row:

$ python predict_match.py path/to/fixture.json

2. Convenience – just pass HomeTeam AwayTeam (and optional date).  The
   script will look up the most recent head-to-head in the historical
   feature table and reuse its engineered features.

$ python predict_match.py "Real Madrid" "Chelsea" 2026-05-21

If no direct H2H meeting exists in the dataset, the script falls back to
picking each team's most recent league match (same home/away side).

3. Ensemble prediction mode (if ensemble models are available):

$ python predict_match.py "Real Madrid" "Chelsea" --ensemble
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "football_predictor.json"
ENSEMBLE_INFO_PATH = PROJECT_ROOT / "models" / "ensemble_info.json"
FEATS_PATH = PROJECT_ROOT / "data" / "training_features.parquet"


class XGBoostPredictor:
    """XGBoost prediction engine with ensemble support."""
    
    def __init__(self, use_ensemble: bool = False):
        self.use_ensemble = use_ensemble
        self.model = None
        self.ensemble_models = {}
        self.ensemble_info = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load XGBoost model(s)."""
        if self.use_ensemble and ENSEMBLE_INFO_PATH.exists():
            # Load ensemble
            with open(ENSEMBLE_INFO_PATH, 'r') as f:
                self.ensemble_info = json.load(f)
            
            model_dir = PROJECT_ROOT / "models"
            for model_path in self.ensemble_info['model_paths']:
                full_path = model_dir / model_path
                if full_path.exists():
                    instance_id = model_path.split('_')[-1].replace('.json', '')
                    self.ensemble_models[instance_id] = xgb.Booster()
                    self.ensemble_models[instance_id].load_model(str(full_path))
            
            print(f"[XGBoost Predictor] Loaded {len(self.ensemble_models)} ensemble models")
        else:
            # Load single model
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            
            self.model = xgb.Booster()
            self.model.load_model(str(MODEL_PATH))
            print(f"[XGBoost Predictor] Loaded single model from {MODEL_PATH}")
    
    def predict(self, df_row: pd.DataFrame) -> dict:
        """Make prediction(s) using loaded model(s)."""
        # Handle categorical features
        for col in df_row.columns:
            if df_row[col].dtype == 'object':
                df_row[col] = df_row[col].astype('category').cat.codes
        
        # Fill NaN values
        df_row = df_row.fillna(0)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(df_row, enable_categorical=False)
        
        if self.use_ensemble and self.ensemble_models:
            return self._predict_ensemble(dmatrix)
        else:
            return self._predict_single(dmatrix)
    
    def _predict_single(self, dmatrix: xgb.DMatrix) -> dict:
        """Single model prediction."""
        probs = self.model.predict(dmatrix)[0]
        return {
            "HomeWin": float(probs[0]),
            "Draw": float(probs[1]),
            "AwayWin": float(probs[2]),
            "model_type": "single"
        }
    
    def _predict_ensemble(self, dmatrix: xgb.DMatrix) -> dict:
        """Ensemble prediction with confidence metrics."""
        predictions = []
        
        for model in self.ensemble_models.values():
            pred = model.predict(dmatrix)[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Average prediction
        avg_pred = np.mean(predictions, axis=0)
        
        # Calculate confidence metrics
        std_dev = np.std(predictions, axis=0)
        confidence = 1.0 - np.mean(std_dev)  # Higher when predictions agree
        
        # Weighted prediction (by training performance if available)
        if 'training_results' in self.ensemble_info:
            weights = []
            for instance_id in self.ensemble_models.keys():
                if instance_id in self.ensemble_info['training_results']:
                    val_loss = self.ensemble_info['training_results'][instance_id]['val_logloss']
                    weights.append(1.0 / val_loss)
                else:
                    weights.append(1.0)
            
            weights = np.array(weights) / sum(weights)
            weighted_pred = np.average(predictions, axis=0, weights=weights)
        else:
            weighted_pred = avg_pred
        
        return {
            "HomeWin": float(weighted_pred[0]),
            "Draw": float(weighted_pred[1]),
            "AwayWin": float(weighted_pred[2]),
            "model_type": "ensemble",
            "confidence": float(confidence),
            "n_models": len(self.ensemble_models),
            "individual_predictions": {
                f"model_{i}": {
                    "HomeWin": float(pred[0]),
                    "Draw": float(pred[1]),
                    "AwayWin": float(pred[2])
                }
                for i, pred in enumerate(predictions)
            }
        }


def _prepare_features_from_h2h(home: str, away: str) -> pd.DataFrame:
    """Get the most recent *engineered* feature row for *home vs away*.

    Falls back to each team's most recent match if the exact pairing is
    missing.  Drops non-model columns and returns a 1-row DataFrame.
    """
    df = pd.read_parquet(FEATS_PATH)

    # try exact most-recent head-to-head first
    mask = (df["HomeTeam"].str.lower() == home.lower()) & (
        df["AwayTeam"].str.lower() == away.lower()
    )
    cand = df[mask].sort_values("Date", ascending=False)

    if cand.empty:
        # fallback → most recent home match of *home* vs anyone
        cand_home = (
            df[df["HomeTeam"].str.lower() == home.lower()]
            .sort_values("Date", ascending=False)
            .head(1)
        )
        cand_away = (
            df[df["AwayTeam"].str.lower() == away.lower()]
            .sort_values("Date", ascending=False)
            .head(1)
        )
        if cand_home.empty or cand_away.empty:
            raise ValueError("Cannot locate recent matches for supplied teams.")

        # merge the two rows by taking home-side engineered columns from
        # cand_home and away-side columns from cand_away
        home_cols = [c for c in df.columns if c.startswith("Home_")]
        away_cols = [c for c in df.columns if c.startswith("Away_")]
        merged = pd.DataFrame({})
        for col in home_cols:
            merged[col] = cand_home.iloc[0][col]
        for col in away_cols:
            merged[col] = cand_away.iloc[0][col]
        cand = merged

    row = cand.head(1).copy()

    # Get feature names from a sample model
    try:
        sample_model = xgb.Booster()
        sample_model.load_model(str(MODEL_PATH))
        needed = sample_model.feature_names
        if needed:
            row = row[[c for c in row.columns if c in needed]]
            # re-order to model order
            row = row[needed]
    except:
        # If we can't load model, just use available columns
        pass
    
    return row


def main():
    parser = argparse.ArgumentParser(description="Predict football match outcomes")
    parser.add_argument("input", nargs="*", help="Team names or JSON file path")
    parser.add_argument("--ensemble", action="store_true", 
                       help="Use ensemble prediction (if available)")
    
    args = parser.parse_args()
    
    if not args.input:
        print(
            "Usage:\n"
            "  python predict_match.py fixture.json\n"
            "  python predict_match.py \"Home Team\" \"Away Team\" [--ensemble]\n"
            "  python predict_match.py \"Home Team\" \"Away Team\" YYYY-MM-DD [--ensemble]",
            file=sys.stderr,
        )
        sys.exit(1)
    
    # Initialize predictor
    try:
        predictor = XGBoostPredictor(use_ensemble=args.ensemble)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    if len(args.input) == 1 and args.input[0].endswith(".json"):
        # legacy JSON-file mode
        fixture_path = Path(args.input[0])
        if not fixture_path.exists():
            print(f"File not found: {fixture_path}", file=sys.stderr)
            sys.exit(1)

        df = pd.read_json(fixture_path, orient="records")
        result = predictor.predict(df)
        
    elif len(args.input) >= 2:
        # shorthand: team names supplied directly
        home_team = args.input[0]
        away_team = args.input[1]
        try:
            df_row = _prepare_features_from_h2h(home_team, away_team)
            result = predictor.predict(df_row)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Invalid arguments", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main() 