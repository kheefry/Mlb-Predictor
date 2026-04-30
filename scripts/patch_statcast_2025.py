"""One-time: join 2025 Statcast columns into games_2025.csv.

Fetches end-of-2025-season Statcast team batting and pitcher data from
Baseball Savant (disk-cached on first run) and adds six columns to
games_2025.csv so train_combined.py can use Statcast features on both years.

The player-team mapping needed to aggregate batter Statcast to team level
is pulled from the MLB Stats API (also cached).

Columns added:
  home_off_xwoba_sc, away_off_xwoba_sc  — team xwOBA (EB-shrunk)
  home_off_barrel_rate, away_off_barrel_rate  — team barrel % (0-100)
  home_sp_xera_sc, away_sp_xera_sc      — starter xERA (EB-shrunk)

Run once after build_dataset_2025.py and before train_combined.py:
    python -m scripts.patch_statcast_2025
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import mlb_api, statcast as sc

GAMES_CSV = ROOT / "data" / "games" / "games_2025.csv"
SEASON_2025 = 2025


def main():
    if not GAMES_CSV.exists():
        print(f"ERROR: {GAMES_CSV} not found. Run scripts.build_dataset_2025 first.")
        return

    print("Loading games_2025.csv...")
    df = pd.read_csv(GAMES_CSV)
    print(f"  {len(df)} rows")

    if "home_off_xwoba_sc" in df.columns:
        print("  Already has Statcast columns — overwriting with fresh 2025 data.")

    print("Fetching 2025 player batting stats (MLB API, cached) for player-team map...")
    bat_stats_2025 = mlb_api.player_season_stats_bulk(SEASON_2025, "hitting", qualified=False)
    player_team_map = {pid: int(s["team_id"])
                       for pid, s in bat_stats_2025.items() if s.get("team_id")}
    print(f"  {len(player_team_map)} player-team entries")

    print("Fetching 2025 Statcast team batting (Baseball Savant, cached after first run)...")
    sc_bat = sc.get_team_batting(SEASON_2025, player_team_map)
    print(f"  {len(sc_bat)} teams")

    print("Fetching 2025 Statcast pitcher stats (Baseball Savant, cached after first run)...")
    sc_pit = sc.get_pitcher_stats(SEASON_2025)
    print(f"  {len(sc_pit)} pitchers")

    def _team_xwoba(tid):
        return sc_bat.get(int(tid) if pd.notna(tid) else -1, {}).get("xwoba", sc.LG_XWOBA)

    def _team_barrel(tid):
        return sc_bat.get(int(tid) if pd.notna(tid) else -1, {}).get("barrel_pct", sc.LG_BARREL_PCT)

    def _sp_xera(sp_id):
        if pd.isna(sp_id):
            return sc.LG_XERA
        return sc.shrunk_pitcher_sc(sc_pit.get(int(sp_id)))["xera_sc"]

    print("Joining Statcast columns...")
    df["home_off_xwoba_sc"]    = df["home_team_id"].apply(_team_xwoba)
    df["away_off_xwoba_sc"]    = df["away_team_id"].apply(_team_xwoba)
    df["home_off_barrel_rate"] = df["home_team_id"].apply(_team_barrel)
    df["away_off_barrel_rate"] = df["away_team_id"].apply(_team_barrel)
    df["home_sp_xera_sc"]      = df["home_sp_id"].apply(_sp_xera)
    df["away_sp_xera_sc"]      = df["away_sp_id"].apply(_sp_xera)

    n_teams_covered = df["home_off_xwoba_sc"].ne(sc.LG_XWOBA).sum()
    print(f"  xwOBA != league-avg in {n_teams_covered}/{len(df)} rows "
          f"(mean={df['home_off_xwoba_sc'].mean():.4f})")
    print(f"  xERA != league-avg in "
          f"{df['home_sp_xera_sc'].ne(sc.LG_XERA).sum()}/{len(df)} rows "
          f"(mean={df['home_sp_xera_sc'].mean():.3f})")

    df.to_csv(GAMES_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {GAMES_CSV.name}")
    print("Next: python -m scripts.build_dataset  (adds Statcast to 2026 CSV)")
    print("Then: python -m scripts.train_combined")


if __name__ == "__main__":
    main()
