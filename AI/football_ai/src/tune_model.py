"""Optuna hyper-parameter search to squeeze extra accuracy out of the LightGBM predictor.

The script performs Bayesian optimisation over ~30 trials (adjust `--trials`) with
5-fold stratified CV.  Objective is multi-class log-loss (lower = better).

Outputs
-------
• `models/football_predictor_best.txt` – best booster found
• `models/optuna_study.db`              – SQLite study to resume / inspect
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "training_features.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "football_predictor_best.txt"
STUDY_DB = MODEL_DIR / "optuna_study.db"

N_THREADS = 10  # use all vCPUs of the VM


def load_data():
    df = pd.read_parquet(DATA_FILE)
    y = df["Result"].astype(int).values
    X = df.drop(columns=["Result", "Date"])
    return X, y


def objective(trial: optuna.trial.Trial):
    X, y = load_data()

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 64, 1024, step=64),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "max_depth": -1,
        "seed": 42,
        "num_threads": N_THREADS,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    losses = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=4000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

        y_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
        losses.append(log_loss(y_val, y_pred))

    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30, help="number of Optuna trials")
    args = parser.parse_args()

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="lgbm_match_predictor",
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("[tune_model] Best multi_logloss:", study.best_value)
    print("[tune_model] Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Train final model on full data with best params
    X, y = load_data()
    dtrain_full = lgb.Dataset(X, label=y)

    best_params = study.best_params
    best_params.update(
        {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "num_threads": N_THREADS,
            "seed": 42,
        }
    )

    booster = lgb.train(
        best_params,
        dtrain_full,
        num_boost_round=study.best_trial.user_attrs.get("best_iter", 1000),
        valid_sets=[dtrain_full],
        callbacks=[lgb.log_evaluation(period=100)],
    )

    booster.save_model(str(BEST_MODEL_PATH))
    print(f"[tune_model] Saved best model → {BEST_MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main() 