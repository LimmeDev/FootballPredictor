# Football-AI – End-to-End Match & Tournament Predictor

Follow the numbered commands in order.  Copy-paste **one block at a time** into your terminal (Linux/macOS or WSL).  Anything that requires a value from you is highlighted in UPPER-CASE.

---
## 1.  Clone / open the project folder
```bash
# From any directory you like – here we stay in your existing workspace
cd ~/Downloads/AI
```

## 2.  Create Python virtual environment (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3.  Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r football_ai/requirements.txt
```

## 4.  Download raw data (open-access; no credentials needed)
```bash
python football_ai/src/download_data.py
```
•  This clones StatsBomb open-data (≈200 MB), downloads FiveThirtyEight SPI CSV (<5 MB), and grabs ~10 seasons of English Premier League stats from Football-Data.co.uk (~1 MB each).

## 5.  Generate feature table
```bash
python football_ai/src/make_features.py
```
The script writes `data/training_features.parquet` with engineered match-level stats.

## 6.  Train baseline model
```bash
python football_ai/src/train_model.py
```
A LightGBM multiclass model is saved under `models/football_predictor.txt`.

## 7.  Predict a future match (example)
Create a minimal JSON file `my_fixture.json` containing **all feature columns** that the model expects.  The easiest way is to copy one row from `data/training_features.parquet`, edit the numeric feature values (or leave as-is for testing), and run:
```bash
python football_ai/src/predict_match.py my_fixture.json
```
The script prints probabilities for Home-win, Draw, Away-win.

## 8.  Next steps
*   Improve feature engineering inside `src/make_features.py` (add rolling averages, Elo, odds).
*   Extend `src/simulate_tournament.py` (template provided) to produce full cup forecasts.

---
### FAQ
1. *"I don't have **git** installed; the StatsBomb clone fails."*
   Run `sudo apt install git -y` and rerun step 4.
2. *"Pandas cannot read a CSV because of encoding."*  Ensure you run python ≥ 3.9 and pandas ≥ 2.0 (already in `requirements.txt`).
3. *"How do I create the JSON for future fixtures?"*  Look at column names in `training_features.parquet`; use zeros where unknown and update the teams' latest rolling metrics.

Happy modelling!  🎉 