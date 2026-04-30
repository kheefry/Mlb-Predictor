"""Build 2025 analytical projection rows for calibrate_reliability.py.

Generates props_bat_2025.csv and props_pit_2025.csv with the same schema as
the 2026 versions, so calibrate_reliability.py can concatenate both years and
compute more stable Pearson r estimates.

All API calls are disk-cached, so this runs in ~30 seconds after
build_dataset_2025.py has run once.

Run: python -m scripts.build_props_2025
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_dataset_2025 import (
    weekly_snapshots, lookup_snapshot,
    SEASON, SEASON_START, SEASON_END,
)
from src import features as feats, model as mdl, parks, projections as proj

GAMES_CSV    = ROOT / "data" / "games" / "games_2025.csv"
BOX_CSV      = ROOT / "data" / "games" / "box_2025.csv"
TEAM_MODEL   = ROOT / "data" / "models" / "team_runs.joblib"
OUT_BAT      = ROOT / "data" / "games" / "props_bat_2025.csv"
OUT_PIT      = ROOT / "data" / "games" / "props_pit_2025.csv"


def _int_keys(d: dict) -> dict[int, dict]:
    return {int(k): v for k, v in d.items()}


def _snap_batter_stats(snap: dict) -> dict[int, dict]:
    """2025 snaps use 'bat' key; 2026 use 'batter_stats'."""
    return _int_keys(snap.get("bat") or snap.get("batter_stats") or {})


def _snap_pitcher_stats(snap: dict) -> dict[int, dict]:
    """2025 snaps use 'pit' key; 2026 use 'pitcher_stats'."""
    return _int_keys(snap.get("pit") or snap.get("pitcher_stats") or {})


def main():
    if not GAMES_CSV.exists():
        print(f"ERROR: {GAMES_CSV} not found. Run scripts.build_dataset_2025 first.")
        return
    if not BOX_CSV.exists():
        print(f"ERROR: {BOX_CSV} not found. Run scripts.build_dataset_2025 first.")
        return
    if not TEAM_MODEL.exists():
        print(f"ERROR: {TEAM_MODEL} not found. Run scripts.train_combined first.")
        return

    print("Loading games_2025.csv and box_2025.csv...")
    games = pd.read_csv(GAMES_CSV)
    box   = pd.read_csv(BOX_CSV)
    final = games[games["is_final"] == True].copy().sort_values("date")
    print(f"  {len(final)} final games, {len(box)} player-game rows")

    print("Loading 2025 weekly snapshots (all cached, ~30 s)...")
    snaps = weekly_snapshots(SEASON, SEASON_START, SEASON_END)
    print(f"  {len(snaps)} snapshots ({min(snaps)} to {max(snaps)})")

    print("Computing per-game predicted runs...")
    team_model = mdl.TeamScoreModel.load(TEAM_MODEL)
    long = mdl.long_form(final)
    long["pred_runs"] = team_model.predict_runs(long)
    pred_by_gs: dict[tuple, float] = {}
    for _, r in long.iterrows():
        side = "home" if r["is_home"] == 1 else "away"
        pred_by_gs[(int(r["game_pk"]), side)] = float(r["pred_runs"])

    print("Building projection rows...")
    bat_rows: list[dict] = []
    pit_rows: list[dict] = []

    for _, g in final.iterrows():
        gpk   = int(g["game_pk"])
        gdate = str(g["date"])
        venue = g["venue"]
        park  = parks.get_park(venue)
        wadj  = {
            "runs_mult":      float(g.get("runs_mult", 1.0) or 1.0),
            "hr_mult":        float(g.get("hr_mult", 1.0) or 1.0),
            "wind_to_cf_mph": float(g.get("wind_to_cf_mph", 0.0) or 0.0),
            "temp_f":         float(g.get("temp_f", 70.0) or 70.0),
        }

        snap = lookup_snapshot(snaps, gdate)
        if snap is None:
            continue

        bat_stats  = _snap_batter_stats(snap)
        pit_stats  = _snap_pitcher_stats(snap)
        team_off   = _int_keys(snap.get("team_off", {}))

        for side in ("home", "away"):
            tid  = int(g["home_team_id"]) if side == "home" else int(g["away_team_id"])
            otid = int(g["away_team_id"]) if side == "home" else int(g["home_team_id"])

            sp_opp_raw = g["away_sp_id"] if side == "home" else g["home_sp_id"]
            sp_opp     = int(sp_opp_raw) if pd.notna(sp_opp_raw) else None

            team_pred = pred_by_gs.get((gpk, side))
            opp_pred  = pred_by_gs.get((gpk, "away" if side == "home" else "home"))
            if team_pred is None or opp_pred is None:
                continue

            opp_sp_q   = feats.pitcher_quality_index(pit_stats.get(sp_opp, {})) if sp_opp else feats.pitcher_quality_index({})
            opp_off_idx = feats.team_offense_index(team_off.get(otid, {}))

            # ---- Batters ----
            actual_b = box[(box["game_pk"] == gpk) & (box["side"] == side) & (box["pa"] > 0)]
            actual_b = actual_b.sort_values("pa", ascending=False).head(9).reset_index(drop=True)

            for order_idx, row in actual_b.iterrows():
                pid = int(row["player_id"])
                bs  = bat_stats.get(pid, {"player_id": pid, "name": row["name"], "team_id": tid})
                # 2025 snaps lack recent/platoon data — pass None/empty
                bproj = proj.project_batter(
                    bs, order_idx + 1, team_pred, opp_sp_q, park, wadj,
                    recent_stats=None, ml_blend=0,
                    bat_side=None, opp_pit_throws=None, bat_split=None,
                )
                bat_rows.append({
                    "proj_h":    bproj.proj_h,
                    "proj_hr":   bproj.proj_hr,
                    "proj_tb":   bproj.proj_tb,
                    "proj_rbi":  bproj.proj_rbi,
                    "proj_runs": bproj.proj_runs,
                    "proj_k":    bproj.proj_k,
                    "proj_bb":   bproj.proj_bb,
                    "expected_pa": bproj.expected_pa,
                    "date":      gdate,
                    "game_pk":   gpk,
                    "side":      side,
                    "player_id": pid,
                    "name":      row["name"],
                    "h":         float(row["h"]),
                    "hr":        float(row["hr"]),
                    "tb":        float(row["tb"]),
                    "rbi":       float(row["rbi"]),
                    "runs_b":    float(row["runs_b"]),
                    "k_b":       float(row["k_b"]),
                    "bb_b":      float(row["bb_b"]),
                    "pa":        float(row["pa"]),
                })

            # ---- Starters ----
            actual_p = box[
                (box["game_pk"] == gpk) &
                (box["side"] == side) &
                (box["started"] == True)
            ]
            for _, prow in actual_p.iterrows():
                pid = int(prow["player_id"])
                ps  = pit_stats.get(pid, {"player_id": pid, "name": prow["name"], "team_id": tid})
                pproj = proj.project_pitcher(
                    ps, tid, opp_off_idx, opp_pred, park, wadj,
                    recent_stats=None, ml_blend=0,
                )
                pit_rows.append({
                    "proj_k":        pproj.proj_k,
                    "proj_bb":       pproj.proj_bb,
                    "proj_h":        pproj.proj_h,
                    "proj_er":       pproj.proj_er,
                    "proj_hr":       pproj.proj_hr_allowed,
                    "expected_outs": pproj.expected_outs,
                    "expected_ip":   pproj.expected_ip,
                    "date":          gdate,
                    "game_pk":       gpk,
                    "side":          side,
                    "player_id":     pid,
                    "name":          prow["name"],
                    "k_p":           float(prow["k_p"]),
                    "outs":          float(prow["outs"]),
                    "er":            float(prow["er"]),
                    "h_p":           float(prow["h_p"]),
                    "bb_p":          float(prow["bb_p"]),
                    "hr_p":          float(prow["hr_p"]),
                })

    bat_df = pd.DataFrame(bat_rows)
    pit_df = pd.DataFrame(pit_rows)

    bat_df.to_csv(OUT_BAT, index=False)
    pit_df.to_csv(OUT_PIT, index=False)

    print(f"\nDone.")
    print(f"  Batter rows:  {len(bat_df):,} -> {OUT_BAT.name}")
    print(f"  Pitcher rows: {len(pit_df):,} -> {OUT_PIT.name}")
    print(f"\nNow run:  python -m scripts.calibrate_reliability")


if __name__ == "__main__":
    main()
