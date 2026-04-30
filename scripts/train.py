"""Train the team scoring model and run a temporal-holdout backtest +
walk-forward CV + calibration analysis.

Run: python -m scripts.train
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model as mdl, cv

GAMES_CSV = ROOT / "data" / "games" / "games_2026.csv"
MODEL_OUT = ROOT / "data" / "models" / "team_runs.joblib"


def main():
    df = pd.read_csv(GAMES_CSV)
    print(f"Loaded {len(df)} games  (final: {(df['is_final']==True).sum()})")

    # ---- Single fit (last 7 days as holdout) ----
    model, eval_df = mdl.fit(df, holdout_days=7, use_gbt=True)
    print("\n=== Single-holdout backtest (last 7 days) ===")
    print(f"Train games: {model.train_games}")
    print(f"Train MAE   : {model.train_mae:.3f}")
    print(f"Test MAE    : {eval_df.attrs['test_mae']:.3f}  (n={len(eval_df)//2} games)")
    print(f"  GLM-only MAE       : {eval_df.attrs['test_mae_glm']:.3f}")
    if eval_df.attrs['test_mae_gbt'] is not None:
        print(f"  GBT-only MAE       : {eval_df.attrs['test_mae_gbt']:.3f}")
        print(f"  Ensemble blend w_glm = {eval_df.attrs['blend_w_glm']:.2f}")
    if "test_mae_calibrated" in eval_df.attrs:
        print(f"  Isotonic-calibrated MAE: {eval_df.attrs['test_mae_calibrated']:.3f}")

    bl_const = float(np.mean(np.abs(eval_df["y_runs"] - eval_df["y_runs"].mean())))
    bl_off = float(np.mean(np.abs(eval_df["y_runs"] - eval_df["off_rpg"])))
    print(f"  baseline (constant mean) MAE: {bl_const:.3f}")
    print(f"  baseline (team RPG)      MAE: {bl_off:.3f}")

    # Game-level totals
    games = (eval_df.groupby("game_pk")
                  .agg(actual_total=("y_runs", "sum"),
                       pred_total=("pred_runs", "sum"))
                  .reset_index())
    print(f"  Game total runs MAE  : {(games['actual_total']-games['pred_total']).abs().mean():.3f}")
    print(f"  Game total runs RMSE : {np.sqrt(((games['actual_total']-games['pred_total'])**2).mean()):.3f}")

    # Save the model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nSaved model -> {MODEL_OUT}")

    # ---- Walk-forward CV ----
    print("\n=== Walk-forward CV (rolling, 7-day folds) ===")
    cv_res = cv.walk_forward_team_runs(df, min_train_days=14, fold_days=7, use_gbt=True)
    if len(cv_res["folds"]):
        folds = cv_res["folds"]
        print(folds.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nMean per-fold MAE:")
        for col in ["MAE_glm", "MAE_gbt", "MAE_ens@prior_w", "MAE_oracle", "MAE_calibrated", "MAE_baseline"]:
            print(f"  {col:18s}  {folds[col].mean():.3f}  (std {folds[col].std():.3f})")

        # Calibration on concatenated CV preds
        preds = cv_res["all_preds"]
        print("\n=== Calibration: raw GLM (predicted bin -> empirical mean) ===")
        cal_raw = cv.calibration_table(preds, lam_col="pred_glm", n_bins=6)
        print(cal_raw.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        print("\n=== Calibration: after isotonic recalibration ===")
        cal_iso = cv.calibration_table(preds, lam_col="pred_calibrated", n_bins=6)
        print(cal_iso.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        # Moneyline accuracy
        ml_raw = cv.moneyline_accuracy(preds, lam_col="pred_glm")
        ml_cal = cv.moneyline_accuracy(preds, lam_col="pred_calibrated")
        print(f"\nMoneyline accuracy (GLM raw)        : {ml_raw['accuracy']:.3f}  (n={ml_raw['n_games']} games)")
        print(f"Moneyline accuracy (GLM calibrated) : {ml_cal['accuracy']:.3f}  (n={ml_cal['n_games']} games)")
        if ml_raw["high_conf_accuracy"] is not None:
            print(f"  Raw high-conf (|edge|>0.7 runs)   : "
                  f"{ml_raw['high_conf_accuracy']:.3f}  (n={ml_raw['n_high_conf']} games)")
    else:
        print("Not enough data for walk-forward CV.")

    # ---- Feature importance ----
    print("\n=== GLM coefficients (z-scored features) ===")
    coefs = dict(zip(mdl.FEATURES, model.glm.coef_))
    for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1]))[:15]:
        print(f"  {k:25s}  {v:+.4f}")
    print(f"  intercept                 {model.glm.intercept_:+.4f}  -> base lambda = {np.exp(model.glm.intercept_):.3f}")

    if model.gbt is not None:
        # Permutation importance on holdout for the GBT
        try:
            from sklearn.inspection import permutation_importance
            X_test = model._normalize(eval_df)[mdl.FEATURES].values
            y_test = eval_df["y_runs"].values
            perm = permutation_importance(model.gbt, X_test, y_test, n_repeats=5,
                                          random_state=0, scoring="neg_mean_absolute_error")
            print("\n=== GBT permutation importance (top 10) ===")
            order = np.argsort(perm.importances_mean)[::-1][:10]
            for i in order:
                print(f"  {mdl.FEATURES[i]:25s}  {perm.importances_mean[i]:+.4f}  "
                      f"(std {perm.importances_std[i]:.4f})")
        except Exception as e:
            print(f"(perm importance skipped: {e})")


if __name__ == "__main__":
    main()
