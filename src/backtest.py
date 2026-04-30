"""Backtest player projections against boxscore actuals.

For each completed game:
  1) load the game features
  2) predict team runs (model)
  3) build batter & pitcher projections from the lineups *that actually played*
  4) compare projection to actual boxscore line

We use lineups extracted from the boxscore — that's the cleanest comparison
of model accuracy on a player level (it removes the noise of guessing the
lineup). For a "real-world" backtest we'd reconstruct guessed-lineup error
too, but lineup guessing is a separate problem.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import features as feats
from . import model as mdl
from . import parks, projections as proj


def load_snapshot(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats_lookup(d: dict) -> dict[int, dict]:
    return {int(k): v for k, v in d.items()}


def backtest_players(
    games_csv: Path,
    box_csv: Path,
    snapshot_path: Path,
    model_path: Path,
    holdout_days: int = 7,
) -> dict:
    """Project every batter/pitcher in held-out games and compare to actuals."""
    games = pd.read_csv(games_csv)
    box = pd.read_csv(box_csv)
    snap = load_snapshot(snapshot_path)

    team_off = _stats_lookup(snap["team_off"])
    team_pit = _stats_lookup(snap["team_pit"])
    pitcher_stats = _stats_lookup(snap["pitcher_stats"])
    batter_stats = _stats_lookup(snap["batter_stats"])
    batter_recent = _stats_lookup(snap.get("batter_stats_recent", {}))
    pitcher_recent = _stats_lookup(snap.get("pitcher_stats_recent", {}))
    bat_vs_l = _stats_lookup(snap.get("bat_vs_l", {}))
    bat_vs_r = _stats_lookup(snap.get("bat_vs_r", {}))
    bat_sides = {int(k): v for k, v in snap.get("bat_sides", {}).items()}
    pit_throws = {int(k): v for k, v in snap.get("pit_throws", {}).items()}

    final = games[games["is_final"] == True].sort_values("date")
    cutoff_dates = sorted(final["date"].unique())
    test_dates = set(cutoff_dates[-holdout_days:])
    test = final[final["date"].isin(test_dates)]

    model = mdl.TeamScoreModel.load(model_path)

    # Predict team runs for held-out games
    long = mdl.long_form(test)
    long["pred_runs"] = model.predict_runs(long)

    # Index for fast lookup
    pred_by_game_side = {}
    for _, r in long.iterrows():
        side = "home" if r["is_home"] == 1 else "away"
        pred_by_game_side[(int(r["game_pk"]), side)] = float(r["pred_runs"])

    # ---- Build per-player projections for each game ----
    bat_records: list[dict] = []
    pit_records: list[dict] = []

    for _, g in test.iterrows():
        gpk = int(g["game_pk"])
        venue = g["venue"]
        park = parks.get_park(venue)
        wadj = {
            "runs_mult": float(g["runs_mult"]),
            "hr_mult":   float(g["hr_mult"]),
            "wind_to_cf_mph": float(g.get("wind_to_cf_mph", 0.0)),
            "temp_f": float(g.get("temp_f", 70.0)),
        }
        for side in ("home", "away"):
            tid  = int(g["home_team_id"])    if side == "home" else int(g["away_team_id"])
            otid = int(g["away_team_id"])    if side == "home" else int(g["home_team_id"])
            sp_id = int(g["home_sp_id"]) if side == "home" and not pd.isna(g["home_sp_id"]) else (
                    int(g["away_sp_id"]) if side == "away" and not pd.isna(g["away_sp_id"]) else None)
            opp_sp_id = int(g["away_sp_id"]) if side == "home" and not pd.isna(g["away_sp_id"]) else (
                        int(g["home_sp_id"]) if side == "away" and not pd.isna(g["home_sp_id"]) else None)

            team_pred = pred_by_game_side.get((gpk, side))
            opp_pred  = pred_by_game_side.get((gpk, "away" if side == "home" else "home"))
            if team_pred is None or opp_pred is None:
                continue

            opp_sp_q = feats.pitcher_quality_index(pitcher_stats.get(opp_sp_id, {})) if opp_sp_id else feats.pitcher_quality_index({})
            opp_off_idx = feats.team_offense_index(team_off.get(otid, {}))

            # Use the actual lineup from the boxscore — batters who logged a PA
            actual = box[(box["game_pk"] == gpk) & (box["side"] == side) & (box["pa"] > 0)]
            actual = actual.sort_values("pa", ascending=False).head(9).reset_index(drop=True)

            for order_idx, row in actual.iterrows():
                pid = int(row["player_id"])
                bs = batter_stats.get(pid, {"player_id": pid, "name": row["name"], "team_id": tid})
                rs = batter_recent.get(pid)
                pl = proj.resolve_platoon(pid, opp_sp_id, bat_sides, pit_throws,
                                          bat_vs_l, bat_vs_r)
                p = proj.project_batter(bs, order_idx + 1, team_pred, opp_sp_q, park, wadj,
                                        recent_stats=rs,
                                        bat_side=pl["bat_side"],
                                        opp_pit_throws=pl["opp_pit_throws"],
                                        bat_split=pl["bat_split"])
                bat_records.append({
                    "game_pk": gpk, "date": g["date"], "side": side, "team_id": tid,
                    "player_id": p.player_id, "name": p.name,
                    "actual_pa": int(row["pa"]),
                    "proj_pa": p.expected_pa,
                    "proj_h": p.proj_h,    "actual_h":  int(row["h"]),
                    "proj_hr": p.proj_hr,  "actual_hr": int(row["hr"]),
                    "proj_2b": p.proj_2b,  "actual_2b": int(row["doubles"]),
                    "proj_tb": p.proj_tb,  "actual_tb": int(row["tb"]),
                    "proj_rbi": p.proj_rbi,"actual_rbi": int(row["rbi"]),
                    "proj_runs": p.proj_runs, "actual_runs": int(row["runs_b"]),
                    "proj_k":  p.proj_k,   "actual_k":  int(row["k_b"]),
                    "proj_bb": p.proj_bb,  "actual_bb": int(row["bb_b"]),
                })

            # Pitcher projection (actual starter from the boxscore)
            actual_starters = box[(box["game_pk"] == gpk) & (box["side"] == side) & (box["started"] == True)]
            for _, prow in actual_starters.iterrows():
                pid = int(prow["player_id"])
                ps = pitcher_stats.get(pid, {"player_id": pid, "name": prow["name"], "team_id": tid})
                rs = pitcher_recent.get(pid)
                pp = proj.project_pitcher(ps, tid, opp_off_idx, opp_pred, park, wadj,
                                          recent_stats=rs)
                pit_records.append({
                    "game_pk": gpk, "date": g["date"], "side": side, "team_id": tid,
                    "player_id": pp.player_id, "name": pp.name,
                    "proj_outs": pp.expected_outs, "actual_outs": int(prow["outs"]),
                    "proj_ip":  pp.expected_ip,   "actual_ip":  float(prow["ip"]),
                    "proj_k":  pp.proj_k,   "actual_k":  int(prow["k_p"]),
                    "proj_bb": pp.proj_bb,  "actual_bb": int(prow["bb_p"]),
                    "proj_h":  pp.proj_h,   "actual_h":  int(prow["h_p"]),
                    "proj_er": pp.proj_er,  "actual_er": int(prow["er"]),
                    "proj_hr": pp.proj_hr_allowed, "actual_hr": int(prow["hr_p"]),
                })

    bat_df = pd.DataFrame(bat_records)
    pit_df = pd.DataFrame(pit_records)
    return {"batters": bat_df, "pitchers": pit_df}


def summarize(df: pd.DataFrame, stats: list[tuple[str, str]]) -> pd.DataFrame:
    """For each (proj_col, actual_col) report MAE, mean projection, mean actual, R^2."""
    rows = []
    for proj_col, actual_col in stats:
        if proj_col not in df.columns or actual_col not in df.columns:
            continue
        d = df[[proj_col, actual_col]].dropna()
        if len(d) == 0:
            continue
        err = d[proj_col] - d[actual_col]
        sse = (err ** 2).sum()
        sst = ((d[actual_col] - d[actual_col].mean()) ** 2).sum() or 1.0
        rows.append({
            "stat": proj_col.replace("proj_", ""),
            "n": len(d),
            "mean_proj": d[proj_col].mean(),
            "mean_actual": d[actual_col].mean(),
            "MAE": err.abs().mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "R2": 1 - sse / sst,
        })
    return pd.DataFrame(rows)


def prop_calibration(df: pd.DataFrame, proj_col: str, actual_col: str,
                     n_bins: int = 5) -> pd.DataFrame:
    """For a given (projection, actual) pair, bin players by projection magnitude
    and report the empirical mean. Tells us whether top-projected players really
    hit higher rates than bottom-projected ones (essential for prop betting).
    """
    d = df[[proj_col, actual_col]].dropna()
    if len(d) == 0:
        return pd.DataFrame()
    d = d.copy()
    d["bin"] = pd.qcut(d[proj_col], q=n_bins, duplicates="drop")
    grp = d.groupby("bin", observed=True).agg(
        n=(actual_col, "size"),
        mean_proj=(proj_col, "mean"),
        mean_actual=(actual_col, "mean"),
    ).reset_index()
    grp["bias"] = grp["mean_proj"] - grp["mean_actual"]
    grp["lift_vs_overall"] = grp["mean_actual"] / d[actual_col].mean()
    return grp
