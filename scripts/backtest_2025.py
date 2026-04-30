"""Full-season 2025 backtest of the team-runs model.

Trains on the first half of 2025, tests on the second half. Reports MAE,
moneyline accuracy, total-runs RMSE, and per-month error so we can spot
systematic biases (e.g. early-season noise dominating).

Run: python -m scripts.backtest_2025
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model as mdl


GAMES = ROOT / "data" / "games" / "games_2025.csv"


def main():
    df = pd.read_csv(GAMES)
    print(f"Loaded {len(df)} 2025 games")

    # Use the standard fit() helper but with longer holdout
    long_total = mdl.long_form(df).dropna(subset=["y_runs"]).copy()
    long_total["y_runs"] = long_total["y_runs"].astype(float)
    long_total = long_total.sort_values("date")

    # Time-series CV: roll forward in monthly folds
    long_total["month"] = pd.to_datetime(long_total["date"]).dt.month
    months = sorted(long_total["month"].unique())
    print(f"Months: {months}")

    # Walk-forward: train on months 1..k, test on month k+1
    fold_results = []
    for split in range(2, len(months)):
        train_months = months[:split]
        test_month = months[split]
        train = long_total[long_total["month"].isin(train_months)]
        test  = long_total[long_total["month"] == test_month]
        if len(train) < 100 or len(test) < 50:
            continue

        # Use feature normalization from train fold only
        means = {c: float(train[c].mean()) for c in mdl.FEATURES if c != "is_home"}
        stds  = {c: float(train[c].std() or 1.0) for c in mdl.FEATURES if c != "is_home"}
        Xtr = mdl._normalize_features(train, means, stds)[mdl.FEATURES].values
        Xte = mdl._normalize_features(test,  means, stds)[mdl.FEATURES].values
        ytr = train["y_runs"].values; yte = test["y_runs"].values

        from sklearn.linear_model import PoissonRegressor
        m = PoissonRegressor(alpha=1.0, max_iter=5000).fit(Xtr, ytr)
        pred = m.predict(Xte)
        mae = float(np.mean(np.abs(pred - yte)))
        bias = float(np.mean(pred - yte))
        fold_results.append({
            "test_month": test_month,
            "n_train": len(train) // 2,
            "n_test_games": len(test) // 2,
            "MAE": mae,
            "bias": bias,
        })

    print("\n=== Walk-forward (train through month K, test month K+1) ===")
    cv = pd.DataFrame(fold_results)
    print(cv.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nMean fold MAE: {cv['MAE'].mean():.3f} (std {cv['MAE'].std():.3f})")
    print(f"Mean fold bias: {cv['bias'].mean():+.3f}")

    # Single split: train Apr-Aug, test Sep
    train = long_total[long_total["month"] <= 8]
    test  = long_total[long_total["month"] == 9]
    if len(test) > 50:
        means = {c: float(train[c].mean()) for c in mdl.FEATURES if c != "is_home"}
        stds  = {c: float(train[c].std() or 1.0) for c in mdl.FEATURES if c != "is_home"}
        Xtr = mdl._normalize_features(train, means, stds)[mdl.FEATURES].values
        Xte = mdl._normalize_features(test,  means, stds)[mdl.FEATURES].values
        ytr = train["y_runs"].values; yte = test["y_runs"].values
        from sklearn.linear_model import PoissonRegressor
        m = PoissonRegressor(alpha=1.0, max_iter=5000).fit(Xtr, ytr)
        pred = m.predict(Xte)
        print(f"\n=== Single split: train Apr-Aug ({len(train)//2} games), test Sep ({len(test)//2} games) ===")
        print(f"Test MAE: {np.mean(np.abs(pred - yte)):.3f}")
        # Total runs
        test = test.copy(); test["pred"] = pred
        # Pair home/away predictions back into game totals
        gt = test.groupby("game_pk").agg(actual=("y_runs", "sum"), pred=("pred", "sum"))
        print(f"Game total MAE: {(gt['actual'] - gt['pred']).abs().mean():.3f}")
        print(f"Game total RMSE: {np.sqrt(((gt['actual'] - gt['pred'])**2).mean()):.3f}")

        # Moneyline accuracy
        h = test[test["is_home"] == 1].set_index("game_pk")[["pred", "y_runs"]]
        a = test[test["is_home"] == 0].set_index("game_pk")[["pred", "y_runs"]]
        j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
        j = j[j["y_runs_h"] != j["y_runs_a"]]
        acc = ((j["pred_h"] > j["pred_a"]) == (j["y_runs_h"] > j["y_runs_a"])).mean()
        print(f"Moneyline pick accuracy: {acc:.3f} (n={len(j)} games)")

        # Bias by predicted-runs decile
        print("\n  -- Bias by predicted-runs bin --")
        test["bin"] = pd.qcut(test["pred"], q=6, duplicates="drop", labels=False)
        cal = test.groupby("bin").agg(
            n=("y_runs", "size"),
            mean_pred=("pred", "mean"),
            mean_actual=("y_runs", "mean"),
        )
        cal["bias"] = cal["mean_pred"] - cal["mean_actual"]
        print(cal.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
