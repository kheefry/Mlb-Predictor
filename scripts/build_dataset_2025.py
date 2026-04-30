"""Build the 2025 historical dataset for full-season backtesting.

Approach: pull every 2025 game + boxscore, but use weekly snapshots of
season-to-date team/player stats so each game's features come from data
that EXISTED before the game was played (proper temporal split, no leakage).

This is more expensive than the 2026 builder (~26 weekly snapshots × 4 stat
groups, plus all boxscores) but only needs to run once — everything is
disk-cached so re-runs are instant.

Run: python -m scripts.build_dataset_2025
"""
from __future__ import annotations
import csv
import json
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import mlb_api, features as feats

SEASON = 2025
SEASON_START = date(2025, 3, 27)        # opening day 2025
SEASON_END = date(2025, 9, 28)          # regular-season end


def weekly_snapshots(season: int, start: date, end: date) -> dict[str, dict]:
    """Pull season-to-date team/player stats for every Monday in [start, end].

    Returns {iso_date: {team_off, team_pit, pit, bat}} keyed by snapshot date.
    Each game uses the most recent snapshot BEFORE its game date.
    """
    snaps: dict[str, dict] = {}
    d = start
    while d <= end:
        if d.weekday() == 0 or d == start:        # Mondays + first day
            print(f"  snap {d.isoformat()}...")
            try:
                team_off = mlb_api.team_stats_by_range(season, "hitting", start, d)
                team_pit = mlb_api.team_stats_by_range(season, "pitching", start, d)
                bat = mlb_api.player_stats_by_range(season, "hitting", start, d)
                pit = mlb_api.player_stats_by_range(season, "pitching", start, d)
                snaps[d.isoformat()] = {
                    "team_off": team_off, "team_pit": team_pit,
                    "bat": bat, "pit": pit,
                }
            except Exception as e:
                print(f"    snap failed: {e}")
        d += timedelta(days=1)
    return snaps


def lookup_snapshot(snaps: dict[str, dict], game_date: str) -> dict | None:
    """Find the snapshot for the most recent Monday on or before game_date."""
    keys = sorted(snaps.keys())
    chosen = None
    for k in keys:
        if k < game_date:
            chosen = k
        else:
            break
    return snaps.get(chosen) if chosen else (snaps[keys[0]] if keys else None)


def main():
    print(f"Pulling 2025 schedule {SEASON_START} - {SEASON_END}...")
    games = mlb_api.schedule_range(SEASON_START, SEASON_END)
    final_games = [g for g in games if (g.get("status") or {}).get("codedGameState") == "F"]
    print(f"  total: {len(games)}  final: {len(final_games)}")

    print(f"Pulling weekly stat snapshots ({(SEASON_END - SEASON_START).days // 7 + 1} weeks)...")
    snaps = weekly_snapshots(SEASON, SEASON_START, SEASON_END)
    print(f"  {len(snaps)} snapshots loaded")

    out_dir = ROOT / "data" / "games"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building feature rows + boxscores ({len(final_games)} final games)...")
    feat_rows: list[dict] = []
    box_rows: list[dict] = []
    for i, g in enumerate(final_games):
        if i % 200 == 0:
            print(f"  ...{i}/{len(final_games)}")
        # Look up the prior-Monday snapshot for this game's stats
        gdate = g.get("officialDate") or g.get("gameDate", "")[:10]
        snap = lookup_snapshot(snaps, gdate)
        if not snap:
            continue
        try:
            f = feats.build_game_features(
                g, snap["team_off"], snap["team_pit"], snap["pit"],
                team_off_recent=None, team_pit_recent=None,
            )
        except Exception:
            continue
        if f is None:
            continue
        feat_rows.append(asdict(f))

        # Boxscore extract for player-level backtest
        try:
            box = mlb_api.boxscore(f.game_pk)
            for side in ("home", "away"):
                pdata = box.get("teams", {}).get(side, {}).get("players", {})
                team_id = f.home_team_id if side == "home" else f.away_team_id
                opp_id = f.away_team_id if side == "home" else f.home_team_id
                for pid_str, pl in pdata.items():
                    person = pl.get("person", {})
                    pos = pl.get("position", {}).get("abbreviation", "")
                    s = pl.get("stats", {})
                    bat = s.get("batting") or {}
                    pit = s.get("pitching") or {}
                    if not bat and not pit:
                        continue
                    if (bat.get("plateAppearances") or 0) == 0 and not pit:
                        continue
                    box_rows.append({
                        "game_pk": f.game_pk, "date": f.date, "venue": f.venue,
                        "side": side, "team_id": team_id, "opp_team_id": opp_id,
                        "player_id": person.get("id"), "name": person.get("fullName", ""),
                        "position": pos,
                        "pa": bat.get("plateAppearances", 0) or 0,
                        "ab": bat.get("atBats", 0) or 0, "h":  bat.get("hits", 0) or 0,
                        "doubles": bat.get("doubles", 0) or 0,
                        "triples": bat.get("triples", 0) or 0,
                        "hr": bat.get("homeRuns", 0) or 0,
                        "rbi": bat.get("rbi", 0) or 0,
                        "runs_b": bat.get("runs", 0) or 0,
                        "bb_b": bat.get("baseOnBalls", 0) or 0,
                        "k_b": bat.get("strikeOuts", 0) or 0,
                        "tb": bat.get("totalBases", 0) or 0,
                        "sb": bat.get("stolenBases", 0) or 0,
                        "ip": feats._ip_to_outs(pit.get("inningsPitched")) / 3.0 if pit else 0.0,
                        "outs": feats._ip_to_outs(pit.get("inningsPitched")) if pit else 0,
                        "h_p": pit.get("hits", 0) or 0,
                        "er":  pit.get("earnedRuns", 0) or 0,
                        "k_p": pit.get("strikeOuts", 0) or 0,
                        "bb_p": pit.get("baseOnBalls", 0) or 0,
                        "hr_p": pit.get("homeRuns", 0) or 0,
                        "bf": pit.get("battersFaced", 0) or 0,
                        "started": (pit.get("gamesStarted", 0) or 0) > 0,
                    })
        except Exception:
            pass

    # Persist
    feat_csv = out_dir / f"games_{SEASON}.csv"
    if feat_rows:
        keys = list(feat_rows[0].keys())
        with feat_csv.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader(); wr.writerows(feat_rows)

    box_csv = out_dir / f"box_{SEASON}.csv"
    if box_rows:
        keys = list(box_rows[0].keys())
        with box_csv.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader(); wr.writerows(box_rows)

    print(f"\nDone.")
    print(f"  feature rows: {len(feat_rows)}")
    print(f"  player-game rows: {len(box_rows)}")
    print(f"  -> {feat_csv}")
    print(f"  -> {box_csv}")


if __name__ == "__main__":
    main()
