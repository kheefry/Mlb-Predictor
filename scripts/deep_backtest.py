"""Comprehensive stat-projection backtest across all completed games.

Tests every batter and pitcher stat projection against boxscore actuals.
Shows:
  - Overall MAE / bias per stat
  - Decile calibration (top-projected vs top-actual)
  - Per-date trend (are we improving?)
  - Top individual misses per stat

Run:  python -m scripts.deep_backtest [--days N]  (default: all final games)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as feats, model as mdl, parks, projections as proj, statcast as sc

GAMES_CSV = ROOT / "data" / "games" / "games_2026.csv"
BOX_CSV   = ROOT / "data" / "games" / "box_2026.csv"
SNAP      = ROOT / "data" / "games" / "snapshot_2026.json"
MODEL_DIR = ROOT / "data" / "models"


# ── helpers ──────────────────────────────────────────────────────────────────
def _lu(d): return {int(k): v for k, v in d.items()}

SEP = "-" * 68


def _banner(title):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")


def _summarize(df, pairs):
    rows = []
    for pc, ac, label in pairs:
        if pc not in df.columns or ac not in df.columns:
            continue
        d = df[[pc, ac]].dropna()
        if len(d) < 5:
            continue
        err = d[pc] - d[ac]
        sst = ((d[ac] - d[ac].mean()) ** 2).sum() or 1.0
        rows.append({
            "stat":        label,
            "n":           len(d),
            "mean_proj":   round(d[pc].mean(), 3),
            "mean_actual": round(d[ac].mean(), 3),
            "bias":        round(err.mean(), 3),
            "MAE":         round(err.abs().mean(), 3),
            "RMSE":        round(np.sqrt((err**2).mean()), 3),
            "R²":          round(1 - (err**2).sum() / sst, 3),
        })
    return pd.DataFrame(rows)


def _decile_cal(df, pc, ac, label, n_bins=10):
    d = df[[pc, ac, "date"]].dropna()
    if len(d) < 20:
        return
    d = d.copy()
    try:
        d["bin"] = pd.qcut(d[pc], q=n_bins, duplicates="drop", labels=False)
    except Exception:
        return
    grp = d.groupby("bin", observed=True).agg(
        n=(ac, "size"),
        mean_proj=(pc, "mean"),
        mean_actual=(ac, "mean"),
    ).reset_index()
    grp["bias"] = (grp["mean_proj"] - grp["mean_actual"]).round(3)
    print(f"\n  {label}  (decile calibration — decile 10 = highest projected)")
    print(f"  {'Decile':>7} {'n':>5} {'Proj':>7} {'Actual':>7} {'Bias':>7}")
    for _, r in grp.iterrows():
        marker = " << top" if r["bin"] == grp["bin"].max() else ""
        print(f"  {int(r['bin'])+1:>7} {int(r['n']):>5} {r['mean_proj']:>7.3f} "
              f"{r['mean_actual']:>7.3f} {r['bias']:>+7.3f}{marker}")


def _top_misses(df, pc, ac, label, n=10):
    d = df[[pc, ac, "name", "date", "game_pk"]].dropna().copy()
    d["err"] = d[pc] - d[ac]
    d["abs_err"] = d["err"].abs()
    worst = d.nlargest(n, "abs_err")
    print(f"\n  {label} — top {n} absolute misses:")
    print(f"  {'Name':<25} {'Date':>10} {'Proj':>7} {'Actual':>7} {'Err':>7}")
    for _, r in worst.iterrows():
        print(f"  {r['name']:<25} {r['date']:>10} {r[pc]:>7.2f} "
              f"{r[ac]:>7.1f} {r['err']:>+7.2f}")


def _by_date(df, pc, ac, label):
    d = df[[pc, ac, "date"]].dropna()
    grp = d.groupby("date").apply(
        lambda g: pd.Series({
            "n":      len(g),
            "bias":   round((g[pc] - g[ac]).mean(), 3),
            "MAE":    round((g[pc] - g[ac]).abs().mean(), 3),
        })
    ).reset_index()
    print(f"\n  {label} — by date")
    print(f"  {'Date':>12} {'n':>5} {'Bias':>7} {'MAE':>7}")
    for _, r in grp.iterrows():
        print(f"  {r['date']:>12} {int(r['n']):>5} {r['bias']:>+7.3f} {r['MAE']:>7.3f}")


# ── main backtest ─────────────────────────────────────────────────────────────
def run(min_date: str | None = None):
    games = pd.read_csv(GAMES_CSV)
    box   = pd.read_csv(BOX_CSV)
    snap  = json.loads(SNAP.read_text(encoding="utf-8"))

    team_off      = _lu(snap["team_off"])
    team_pit      = _lu(snap["team_pit"])
    pitcher_stats = _lu(snap["pitcher_stats"])
    batter_stats  = _lu(snap["batter_stats"])
    bat_recent    = _lu(snap.get("batter_stats_recent", {}))
    pit_recent    = _lu(snap.get("pitcher_stats_recent", {}))
    bat_vs_l      = _lu(snap.get("bat_vs_l", {}))
    bat_vs_r      = _lu(snap.get("bat_vs_r", {}))
    bat_sides     = {int(k): v for k, v in snap.get("bat_sides", {}).items()}
    pit_throws    = {int(k): v for k, v in snap.get("pit_throws", {}).items()}

    try:
        sc_bat = sc.get_batter_stats(2026)
        sc_pit = sc.get_pitcher_stats(2026)
    except Exception:
        sc_bat, sc_pit = {}, {}

    model    = mdl.TeamScoreModel.load(MODEL_DIR / "team_runs.joblib")
    boot     = mdl.load_bootstrap_ensemble(MODEL_DIR)
    temporal = mdl.load_temporal_ensemble(MODEL_DIR)
    all_models = [model] + boot + temporal

    final = games[games["is_final"] == True].copy()
    if min_date:
        final = final[final["date"] >= min_date]
    final = final.sort_values("date")
    print(f"Testing {len(final)} game-sides across {final['date'].nunique()} dates "
          f"({final['date'].min()} to {final['date'].max()})")

    long = mdl.long_form(final)
    long["pred_runs"] = mdl.predict_ensemble(all_models, long)
    pred_map = {}
    for _, r in long.iterrows():
        side = "home" if r["is_home"] == 1 else "away"
        pred_map[(int(r["game_pk"]), side)] = float(r["pred_runs"])

    bat_rows, pit_rows = [], []

    for _, g in final.drop_duplicates("game_pk").iterrows():
        gpk  = int(g["game_pk"])
        park = parks.get_park(g["venue"])
        wadj = {
            "runs_mult":      float(g["runs_mult"]),
            "hr_mult":        float(g["hr_mult"]),
            "wind_to_cf_mph": float(g.get("wind_to_cf_mph", 0.0)),
            "temp_f":         float(g.get("temp_f", 70.0)),
        }

        for side in ("home", "away"):
            tid  = int(g["home_team_id"]) if side == "home" else int(g["away_team_id"])
            otid = int(g["away_team_id"]) if side == "home" else int(g["home_team_id"])

            def _sp(col):
                v = g.get(col)
                return int(v) if v and not pd.isna(v) else None

            sp_id     = _sp("home_sp_id") if side == "home" else _sp("away_sp_id")
            opp_sp_id = _sp("away_sp_id") if side == "home" else _sp("home_sp_id")

            team_pred = pred_map.get((gpk, side))
            opp_pred  = pred_map.get((gpk, "away" if side == "home" else "home"))
            if team_pred is None or opp_pred is None:
                continue

            opp_sp_q  = feats.pitcher_quality_index(
                pitcher_stats.get(opp_sp_id, {}), sc_stats=sc_pit.get(opp_sp_id)
            ) if opp_sp_id else feats.pitcher_quality_index({})
            opp_off   = feats.team_offense_index(team_off.get(otid, {}))

            # ── batters ──
            actual_bat = box[
                (box["game_pk"] == gpk) & (box["side"] == side) & (box["pa"] > 0)
            ].sort_values("pa", ascending=False).head(9).reset_index(drop=True)

            for order_idx, row in actual_bat.iterrows():
                pid = int(row["player_id"])
                bs  = batter_stats.get(pid, {"player_id": pid, "name": row["name"], "team_id": tid})
                rs  = bat_recent.get(pid)
                pl  = proj.resolve_platoon(pid, opp_sp_id, bat_sides, pit_throws, bat_vs_l, bat_vs_r)
                p   = proj.project_batter(
                    bs, order_idx + 1, team_pred, opp_sp_q, park, wadj,
                    recent_stats=rs,
                    bat_side=pl["bat_side"], opp_pit_throws=pl["opp_pit_throws"],
                    bat_split=pl["bat_split"], is_switch=pl.get("is_switch", False),
                    sc_stats=sc_bat.get(pid),
                )
                bat_rows.append({
                    "game_pk": gpk, "date": g["date"], "side": side,
                    "player_id": pid, "name": p.name, "order": order_idx + 1,
                    "proj_pa": p.expected_pa,   "actual_pa":   int(row["pa"]),
                    "proj_h":  p.proj_h,         "actual_h":    int(row["h"]),
                    "proj_hr": p.proj_hr,        "actual_hr":   int(row["hr"]),
                    "proj_tb": p.proj_tb,        "actual_tb":   int(row["tb"]),
                    "proj_rbi":p.proj_rbi,       "actual_rbi":  int(row["rbi"]),
                    "proj_runs":p.proj_runs,     "actual_runs": int(row["runs_b"]),
                    "proj_k":  p.proj_k,         "actual_k":    int(row["k_b"]),
                    "proj_bb": p.proj_bb,        "actual_bb":   int(row["bb_b"]),
                })

            # ── pitchers ──
            actual_pit = box[
                (box["game_pk"] == gpk) & (box["side"] == side) & (box["started"] == True)
            ]
            for _, prow in actual_pit.iterrows():
                pid = int(prow["player_id"])
                ps  = pitcher_stats.get(pid, {"player_id": pid, "name": prow["name"], "team_id": tid})
                rs  = pit_recent.get(pid)
                pp  = proj.project_pitcher(
                    ps, tid, opp_off, opp_pred, park, wadj,
                    recent_stats=rs, sc_stats=sc_pit.get(pid),
                )
                pit_rows.append({
                    "game_pk": gpk, "date": g["date"], "side": side,
                    "player_id": pid, "name": pp.name,
                    "proj_outs": pp.expected_outs, "actual_outs": int(prow["outs"]),
                    "proj_ip":   pp.expected_ip,   "actual_ip":   float(prow["ip"]),
                    "proj_k":    pp.proj_k,         "actual_k":    int(prow["k_p"]),
                    "proj_bb":   pp.proj_bb,        "actual_bb":   int(prow["bb_p"]),
                    "proj_h":    pp.proj_h,         "actual_h":    int(prow["h_p"]),
                    "proj_er":   pp.proj_er,        "actual_er":   int(prow["er"]),
                    "proj_hr":   pp.proj_hr_allowed,"actual_hr":   int(prow["hr_p"]),
                })

    bat = pd.DataFrame(bat_rows)
    pit = pd.DataFrame(pit_rows)

    print(f"\nBatter player-games:  {len(bat)}")
    print(f"Pitcher starts:       {len(pit)}")

    # ════════════════════════════════════════════════════════
    _banner("BATTER PROJECTIONS — overall accuracy")
    bat_pairs = [
        ("proj_h",    "actual_h",    "hits"),
        ("proj_hr",   "actual_hr",   "HR"),
        ("proj_tb",   "actual_tb",   "TB"),
        ("proj_rbi",  "actual_rbi",  "RBI"),
        ("proj_runs", "actual_runs", "runs"),
        ("proj_k",    "actual_k",    "K (batter)"),
        ("proj_bb",   "actual_bb",   "BB"),
        ("proj_pa",   "actual_pa",   "PA"),
    ]
    print(_summarize(bat, bat_pairs).to_string(index=False))

    _banner("PITCHER PROJECTIONS — overall accuracy")
    pit_pairs = [
        ("proj_outs", "actual_outs", "outs"),
        ("proj_ip",   "actual_ip",   "IP"),
        ("proj_k",    "actual_k",    "K (pitcher)"),
        ("proj_bb",   "actual_bb",   "BB"),
        ("proj_h",    "actual_h",    "H allowed"),
        ("proj_er",   "actual_er",   "ER"),
        ("proj_hr",   "actual_hr",   "HR allowed"),
    ]
    print(_summarize(pit, pit_pairs).to_string(index=False))

    # ════════════════════════════════════════════════════════
    _banner("DECILE CALIBRATION — batter stats")
    print("  Does the model correctly rank players within a game?")
    print("  Positive bias = model over-projects; Negative = under-projects")
    for pc, ac, label in bat_pairs:
        _decile_cal(bat, pc, ac, label, n_bins=5)

    _banner("DECILE CALIBRATION — pitcher stats")
    for pc, ac, label in pit_pairs[:5]:   # outs, IP, K, BB, H
        _decile_cal(pit, pc, ac, label, n_bins=4)

    # ════════════════════════════════════════════════════════
    _banner("BY-DATE TREND — are projections improving?")
    print("\n  BATTERS")
    for pc, ac, label in [("proj_h","actual_h","hits"), ("proj_runs","actual_runs","runs"),
                           ("proj_k","actual_k","K")]:
        _by_date(bat, pc, ac, label)
    print("\n  PITCHERS")
    for pc, ac, label in [("proj_k","actual_k","K"), ("proj_outs","actual_outs","outs"),
                           ("proj_er","actual_er","ER")]:
        _by_date(pit, pc, ac, label)

    # ════════════════════════════════════════════════════════
    _banner("WORST INDIVIDUAL MISSES")
    print("\n  BATTERS")
    for pc, ac, label in [("proj_h","actual_h","hits"), ("proj_tb","actual_tb","TB"),
                           ("proj_runs","actual_runs","runs"), ("proj_hr","actual_hr","HR")]:
        _top_misses(bat, pc, ac, label, n=8)
    print("\n  PITCHERS")
    for pc, ac, label in [("proj_k","actual_k","K"), ("proj_outs","actual_outs","outs"),
                           ("proj_er","actual_er","ER")]:
        _top_misses(pit, pc, ac, label, n=8)

    # ════════════════════════════════════════════════════════
    _banner("SYSTEMIC BIAS CHECK — top vs bottom half")
    print("\n  BATTERS: do top-order hitters get over-projected relative to bottom-order?")
    top5  = bat[bat["order"] <= 5]
    bot4  = bat[bat["order"] >  5]
    for pc, ac, label in [("proj_h","actual_h","hits"), ("proj_runs","actual_runs","runs")]:
        if pc not in bat.columns: continue
        t_bias = (top5[pc] - top5[ac]).mean()
        b_bias = (bot4[pc] - bot4[ac]).mean()
        print(f"    {label}: top-5 bias {t_bias:+.3f}  |  bottom-4 bias {b_bias:+.3f}")

    print("\n  PITCHERS: starters who went deep vs short")
    long_out = pit[pit["actual_outs"] >= 18]   # 6+ IP
    short_out = pit[pit["actual_outs"] < 12]   # < 4 IP
    for pc, ac, label in [("proj_outs","actual_outs","outs"), ("proj_k","actual_k","K")]:
        if pc not in pit.columns: continue
        l_bias = (long_out[pc]  - long_out[ac]).mean()  if len(long_out)  else float("nan")
        s_bias = (short_out[pc] - short_out[ac]).mean() if len(short_out) else float("nan")
        print(f"    {label}: long-outing bias {l_bias:+.3f}  |  short-outing bias {s_bias:+.3f}")

    return bat, pit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-date", default="2026-04-30",
                    help="Earliest game date to include (YYYY-MM-DD)")
    args = ap.parse_args()
    run(min_date=args.from_date)
