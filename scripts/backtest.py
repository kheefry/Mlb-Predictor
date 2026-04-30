"""Run the player-projection backtest and print stat-by-stat accuracy.

Run: python -m scripts.backtest
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import backtest as bt

GAMES = ROOT / "data" / "games" / "games_2026.csv"
BOX   = ROOT / "data" / "games" / "box_2026.csv"
SNAP  = ROOT / "data" / "games" / "snapshot_2026.json"
MODEL = ROOT / "data" / "models" / "team_runs.joblib"


def main():
    res = bt.backtest_players(GAMES, BOX, SNAP, MODEL, holdout_days=7)

    bat = res["batters"]; pit = res["pitchers"]
    print(f"\nBatter projections: {len(bat)} player-games")
    bat_summary = bt.summarize(bat, [
        ("proj_h", "actual_h"),
        ("proj_hr", "actual_hr"),
        ("proj_2b", "actual_2b"),
        ("proj_tb", "actual_tb"),
        ("proj_rbi", "actual_rbi"),
        ("proj_runs", "actual_runs"),
        ("proj_k", "actual_k"),
        ("proj_bb", "actual_bb"),
        ("proj_pa", "actual_pa"),
    ])
    print(bat_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nPitcher projections: {len(pit)} starts")
    pit_summary = bt.summarize(pit, [
        ("proj_outs", "actual_outs"),
        ("proj_ip", "actual_ip"),
        ("proj_k", "actual_k"),
        ("proj_bb", "actual_bb"),
        ("proj_h", "actual_h"),
        ("proj_er", "actual_er"),
        ("proj_hr", "actual_hr"),
    ])
    print(pit_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Per-stat calibration: do top-projected batters actually outperform?
    print(f"\n=== Batter prop calibration (5 bins, are top-projected batters really hitting more?) ===")
    for proj_col, actual_col, label in [
        ("proj_h", "actual_h", "HITS"),
        ("proj_hr", "actual_hr", "HR"),
        ("proj_tb", "actual_tb", "TOTAL BASES"),
        ("proj_k", "actual_k", "STRIKEOUTS"),
    ]:
        cal = bt.prop_calibration(bat, proj_col, actual_col, n_bins=5)
        if len(cal):
            print(f"\n  -- {label} --")
            print("  " + cal.to_string(index=False, float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))

    print(f"\n=== Pitcher prop calibration ===")
    for proj_col, actual_col, label in [
        ("proj_k", "actual_k", "STRIKEOUTS"),
        ("proj_outs", "actual_outs", "OUTS RECORDED"),
        ("proj_er", "actual_er", "EARNED RUNS"),
    ]:
        cal = bt.prop_calibration(pit, proj_col, actual_col, n_bins=4)
        if len(cal):
            print(f"\n  -- {label} --")
            print("  " + cal.to_string(index=False, float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))


if __name__ == "__main__":
    main()
