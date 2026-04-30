"""Walk-forward cross-validation and calibration analysis.

For time series like a baseball season, k-fold CV is wrong — it leaks future
into the past. The honest measure is to roll forward through the season:
train on weeks 1..k, predict week k+1, advance, repeat. This mimics how the
model would have been used live.

We report:
  - Per-fold MAE / RMSE / R^2 for GLM, GBT, ensemble
  - Total runs MAE on game-level (sum the two team predictions)
  - Moneyline accuracy
  - Calibration: empirical hit rate of predicted probabilities, in 10% bins
"""
from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from .model import FEATURES, long_form, _normalize_features


def walk_forward_team_runs(games_df: pd.DataFrame,
                            min_train_days: int = 14,
                            fold_days: int = 7,
                            glm_alpha: float = 1.0,
                            use_gbt: bool = True) -> dict:
    """Run walk-forward CV. Each fold trains on all games before the fold-start
    date and tests on a `fold_days`-window starting at that date.

    Returns a dict with per-fold metrics + concatenated predictions for
    calibration analysis.
    """
    final = games_df[games_df["is_final"] == True].copy()
    long = long_form(final).dropna(subset=["y_runs"]).copy()
    long["y_runs"] = long["y_runs"].astype(float)
    long = long.sort_values("date").reset_index(drop=True)

    dates_sorted = sorted(long["date"].unique())
    if len(dates_sorted) < min_train_days + fold_days:
        return {"folds": [], "all_preds": pd.DataFrame()}

    folds = []
    all_preds = []
    fold_idx = 0
    cur = min_train_days
    while cur < len(dates_sorted):
        end = min(cur + fold_days, len(dates_sorted))
        train_dates = set(dates_sorted[:cur])
        test_dates  = set(dates_sorted[cur:end])
        train = long[long["date"].isin(train_dates)]
        test  = long[long["date"].isin(test_dates)]
        if len(train) < 50 or len(test) == 0:
            cur = end
            continue

        means = {c: float(train[c].mean()) for c in FEATURES if c != "is_home"}
        stds  = {c: float(train[c].std() or 1.0) for c in FEATURES if c != "is_home"}

        Xtr = _normalize_features(train, means, stds)[FEATURES].values
        ytr = train["y_runs"].values
        Xte = _normalize_features(test,  means, stds)[FEATURES].values
        yte = test["y_runs"].values

        glm = PoissonRegressor(alpha=glm_alpha, max_iter=5000)
        glm.fit(Xtr, ytr)
        p_glm = glm.predict(Xte)

        if use_gbt:
            gbt = HistGradientBoostingRegressor(
                loss="poisson", learning_rate=0.05, max_iter=300,
                max_depth=4, min_samples_leaf=20, l2_regularization=1.0,
                random_state=0,
            )
            gbt.fit(Xtr, ytr)
            p_gbt = np.clip(gbt.predict(Xte), 0.0, None)
        else:
            p_gbt = None

        # Pick blend weight on prior fold (cleaner than fitting on test).
        # Simple heuristic: fixed 0.5 unless prior fold's tuned weight available.
        prior = folds[-1] if folds else None
        w = prior["best_w"] if prior else 0.5
        if p_gbt is not None:
            # Re-tune w on this fold for diagnostic only (we report ensemble MAE
            # at the prior weight to avoid leakage).
            best_w, best_mae = w, float("inf")
            for ww in np.linspace(0, 1, 11):
                m = float(np.mean(np.abs(ww * p_glm + (1 - ww) * p_gbt - yte)))
                if m < best_mae:
                    best_mae, best_w = m, float(ww)
            blend_at_w = w * p_glm + (1 - w) * p_gbt
            blend_oracle = best_w * p_glm + (1 - best_w) * p_gbt
        else:
            best_w = 1.0
            blend_at_w = p_glm
            blend_oracle = p_glm

        mae_glm = float(np.mean(np.abs(p_glm - yte)))
        mae_gbt = float(np.mean(np.abs((p_gbt if p_gbt is not None else p_glm) - yte)))
        mae_ens = float(np.mean(np.abs(blend_at_w - yte)))
        mae_oracle = float(np.mean(np.abs(blend_oracle - yte)))

        # Per-fold isotonic calibration: fit on a tail of the training set
        # (last 15%), apply to the test fold. No leakage from test.
        cut = int(len(ytr) * 0.85)
        if cut > 50 and (len(ytr) - cut) >= 20:
            Xa, Xb = Xtr[:cut], Xtr[cut:]
            ya, yb = ytr[:cut], ytr[cut:]
            glm_a = PoissonRegressor(alpha=glm_alpha, max_iter=5000).fit(Xa, ya)
            raw_b = glm_a.predict(Xb)
            iso = IsotonicRegression(out_of_bounds="clip").fit(raw_b, yb)
            p_cal = iso.predict(p_glm)
            mae_cal = float(np.mean(np.abs(p_cal - yte)))
        else:
            p_cal = p_glm
            mae_cal = mae_glm

        # Baseline: team's season RPG
        mae_base = float(np.mean(np.abs(test["off_rpg"].values - yte)))

        folds.append({
            "fold": fold_idx,
            "test_start": dates_sorted[cur],
            "test_end": dates_sorted[end - 1],
            "n_train_obs": len(train),
            "n_test_games": len(test) // 2,
            "MAE_glm": mae_glm,
            "MAE_gbt": mae_gbt,
            "MAE_ens@prior_w": mae_ens,
            "MAE_oracle": mae_oracle,
            "MAE_calibrated": mae_cal,
            "MAE_baseline": mae_base,
            "best_w": best_w,
        })

        out = test.copy()
        out["pred_glm"] = p_glm
        out["pred_calibrated"] = p_cal
        if p_gbt is not None:
            out["pred_gbt"] = p_gbt
            out["pred_ens"] = blend_at_w
            out["pred_oracle"] = blend_oracle
        else:
            out["pred_gbt"] = p_glm
            out["pred_ens"] = p_glm
            out["pred_oracle"] = p_glm
        out["fold"] = fold_idx
        all_preds.append(out)

        fold_idx += 1
        cur = end

    return {
        "folds": pd.DataFrame(folds),
        "all_preds": pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(),
    }


def calibration_table(predictions: pd.DataFrame, lam_col: str = "pred_ens",
                      n_bins: int = 8) -> pd.DataFrame:
    """Bin predictions by predicted runs and report the empirical mean.

    Well-calibrated mean predictions should match observed mean runs in each bin.
    """
    if len(predictions) == 0:
        return pd.DataFrame()
    df = predictions[[lam_col, "y_runs"]].dropna().copy()
    df["bin"] = pd.qcut(df[lam_col], q=n_bins, duplicates="drop")
    grp = df.groupby("bin", observed=True).agg(
        n=("y_runs", "size"),
        mean_pred=(lam_col, "mean"),
        mean_actual=("y_runs", "mean"),
        std_actual=("y_runs", "std"),
    ).reset_index()
    grp["bias"] = grp["mean_pred"] - grp["mean_actual"]
    return grp


def moneyline_accuracy(predictions: pd.DataFrame, lam_col: str = "pred_ens") -> dict:
    """Pick the side with higher predicted runs; report hit rate vs actual winners."""
    home = predictions[predictions["is_home"] == 1].set_index("game_pk")
    away = predictions[predictions["is_home"] == 0].set_index("game_pk")
    j = home.join(away, lsuffix="_h", rsuffix="_a", how="inner")
    j = j[j["y_runs_h"] != j["y_runs_a"]]
    pred_home_win = j[f"{lam_col}_h"] > j[f"{lam_col}_a"]
    actual_home_win = j["y_runs_h"] > j["y_runs_a"]
    acc = (pred_home_win == actual_home_win).mean()
    # Confidence buckets
    pred_diff = (j[f"{lam_col}_h"] - j[f"{lam_col}_a"]).abs()
    high_conf = pred_diff > 0.7
    return {
        "n_games": len(j),
        "accuracy": float(acc),
        "n_high_conf": int(high_conf.sum()),
        "high_conf_accuracy": float((pred_home_win[high_conf] == actual_home_win[high_conf]).mean()) if high_conf.any() else None,
    }
