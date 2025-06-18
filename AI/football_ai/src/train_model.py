"""Train a multiclass LightGBM model to predict match outcomes.

Outputs
-------
models/football_predictor.txt – LightGBM booster file
"""
from __future__ import annotations

import shutil
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "training_features.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "football_predictor.txt"


if __name__ == "__main__":
    print("[train_model] Loading features …")
    df = pd.read_parquet(DATA_FILE)

    print("[train_model] Splitting train/validation …")
    X = df.drop(columns=["Result", "Date"])
    y = df["Result"].astype(int)

    # Time-ordered split prevents look-ahead bias
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)

    params = dict(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        learning_rate=0.03,
        num_leaves=512,
        max_depth=-1,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=0.1,
        lambda_l2=0.2,
        num_threads=10,
        seed=42,
        device_type="gpu",
    )

    print("[train_model] Training LightGBM …")
    booster = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=4000,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
    )

    logloss = log_loss(y_val, booster.predict(X_val))
    print(f"[train_model] Validation log-loss: {logloss:.5f}")

    booster.save_model(str(MODEL_PATH))
    print(f"[train_model] Saved model → {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    # Lightweight export copy for inference-only devices
    shutil.copy(MODEL_PATH, MODEL_DIR / "football_predictor_inference.txt") 