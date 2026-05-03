"""Log top-confidence picks and evaluate their outcomes against actuals.

Usage:
  # In predict_core.py, after building the slate:
  from . import bet_tracker
  bet_tracker.log_picks(target_date, top_value_bets, top_n=10)

  # In app.py Track Record tab:
  from src import bet_tracker
  record = bet_tracker.get_track_record(days=30)
"""
from __future__ import annotations
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "bets" / "bet_log.json"
GAMES_CSV = ROOT / "data" / "games" / "games_2026.csv"
BOX_CSV   = ROOT / "data" / "games" / "box_2026.csv"


# ---------- Logging ----------

def _is_duplicate(entry: dict, existing: list[dict], date_str: str) -> bool:
    """Return True if an equivalent bet is already logged for this date.

    Rules:
    - Same description on same date is always a duplicate.
    - For game-line markets (total, moneyline, run_line): block any bet on the
      same game_pk + market, regardless of direction (Over vs Under).
      This prevents the app being run twice in a day from logging conflicting
      sides of the same game total.
    - For player props: block same game_pk + market + player_id combination.
    """
    game_pk = entry.get("game_pk")
    market  = entry.get("market", "")
    pid     = entry.get("player_id")

    for e in existing:
        if e.get("date") != date_str:
            continue
        # Always block exact description match
        if e.get("description") == entry.get("description"):
            return True
        # Game-line markets: one side per game per day
        if market in ("moneyline", "total", "run_line") and game_pk is not None:
            if e.get("game_pk") == game_pk and e.get("market") == market:
                return True
        # Player props: one bet per player per market per game per day
        elif pid is not None and game_pk is not None:
            if (e.get("game_pk") == game_pk
                    and e.get("market") == market
                    and e.get("player_id") == pid):
                return True
    return False


def log_picks(target_date: str | date, bets: list[dict], top_n: int = 10) -> None:
    """Save the top_n highest-confidence bets to the log file.

    Bets are already sorted by score/confidence in predict_core before being
    passed here. We record each bet's model probability, odds, line, and
    market so we can evaluate outcomes later.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = _load_log()

    date_str = target_date.isoformat() if isinstance(target_date, date) else target_date
    # Sort by confidence descending and take top N
    picks = sorted(bets, key=lambda x: -(x.get("confidence") or 0))[:top_n]

    for p in picks:
        entry = {
            "date":        date_str,
            "logged_at":   datetime.now(timezone.utc).isoformat(),
            "description": p.get("description", ""),
            "market":      p.get("market", ""),
            "line":        p.get("line", 0.0),
            "odds":        p.get("odds", 0),
            "model_prob":  p.get("model_prob", 0.0),
            "novig_prob":  p.get("novig_prob", 0.0),
            "edge_pct":    p.get("edge_pct", 0.0),
            "confidence":  p.get("confidence", 0.0),
            "score":       p.get("score", 0.0),
            "game_pk":     p.get("game_pk"),
            "player_id":   p.get("player_id"),
            "outcome":     None,   # filled by evaluate_outcomes()
            "actual":      None,   # actual stat value or score
        }
        if not _is_duplicate(entry, existing, date_str):
            existing.append(entry)

    LOG_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def _load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    try:
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


# ---------- Outcome evaluation ----------

def evaluate_outcomes() -> int:
    """Fill in outcome fields for any logged picks whose games are now final.

    Returns the number of picks updated.
    Reads games_2026.csv and box_2026.csv. Only fills in picks where
    outcome is still None and the game date has passed.
    """
    entries = _load_log()
    if not entries:
        return 0

    try:
        import pandas as pd
        games = pd.read_csv(GAMES_CSV) if GAMES_CSV.exists() else None
        box   = pd.read_csv(BOX_CSV)   if BOX_CSV.exists()   else None
    except Exception:
        return 0

    today = datetime.now(timezone.utc).date()
    updated = 0

    for e in entries:
        if e.get("outcome") is not None:
            continue
        if e["date"] >= today.isoformat():
            continue  # game hasn't been played yet

        result = _resolve_outcome(e, games, box)
        if result is not None:
            e["outcome"] = result["outcome"]
            e["actual"]  = result["actual"]
            updated += 1

    if updated:
        LOG_PATH.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")

    return updated


def _resolve_outcome(entry: dict, games, box) -> dict | None:
    """Determine win/loss for one logged bet. Returns {outcome, actual} or None."""
    market   = entry.get("market", "")
    desc     = entry.get("description", "").lower()
    line     = float(entry.get("line") or 0.0)
    game_pk  = entry.get("game_pk")
    player_id = entry.get("player_id")

    # --- Game line markets ---
    if market in ("moneyline", "total", "run_line"):
        if games is None or game_pk is None:
            return None
        row = games[games["game_pk"] == int(game_pk)]
        if row.empty or not row.iloc[0].get("is_final", False):
            return None
        r = row.iloc[0]
        home_s = float(r.get("home_score", 0) or 0)
        away_s = float(r.get("away_score", 0) or 0)
        total  = home_s + away_s

        if market == "total":
            side = "over" if "over" in desc else "under"
            actual = total
            if side == "over":
                won = total > line
            else:
                won = total < line
        elif market == "moneyline":
            home_team = str(r.get("home_team", "")).lower()
            away_team = str(r.get("away_team", "")).lower()
            if home_team and home_team in desc:
                won = home_s > away_s
            elif away_team and away_team in desc:
                won = away_s > home_s
            else:
                return None
            actual = f"{away_s:.0f}-{home_s:.0f}"
        elif market == "run_line":
            home_team = str(r.get("home_team", "")).lower()
            away_team = str(r.get("away_team", "")).lower()
            margin = home_s - away_s
            if home_team and home_team in desc:
                won = (margin + line) > 0
                actual = margin
            elif away_team and away_team in desc:
                won = (-margin + (-line)) > 0
                actual = -margin
            else:
                return None
        else:
            return None
        return {"outcome": "W" if won else "L", "actual": actual}

    # --- Player prop markets ---
    if box is None or player_id is None:
        return None

    prows = box[box["player_id"] == int(player_id)]
    # Filter to the specific game if we have game_pk
    if game_pk is not None:
        prows = prows[prows["game_pk"] == int(game_pk)]
    if prows.empty:
        return None

    r = prows.iloc[0]
    stat_map = {
        "prop_hr":           ("hr",    "h"),
        "prop_hits":         ("h",     "h"),
        "prop_tb":           ("tb",    "h"),
        "prop_rbi":          ("rbi",   "h"),
        "prop_runs":         ("runs_b","h"),
        "prop_k":            ("k_b",   "h"),
        "prop_bb":           ("bb_b",  "h"),
        "prop_pitcher_k":    ("k_p",   "h"),
        "prop_pitcher_outs": ("outs",  "h"),
        "prop_pitcher_er":   ("er",    "h"),
        "prop_pitcher_h":    ("h_p",   "h"),
        "prop_pitcher_bb":   ("bb_p",  "h"),
        "prop_pitcher_hr":   ("hr_p",  "h"),
    }
    if market not in stat_map:
        return None

    col, _ = stat_map[market]
    if col not in r:
        return None
    actual = float(r[col] or 0)
    side = "over" if "over" in desc else "under"
    if side == "over":
        won = actual > line
    else:
        won = actual < line
    return {"outcome": "W" if won else "L", "actual": actual}


# ---------- Track record summary ----------

def get_track_record(days: int = 30) -> dict:
    """Return a summary dict for the last `days` days of logged picks.

    Keys:
      total, wins, losses, pending, win_rate, by_market
    """
    entries = _load_log()
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    recent = [e for e in entries if e["date"] >= cutoff]

    total   = len(recent)
    wins    = sum(1 for e in recent if e.get("outcome") == "W")
    losses  = sum(1 for e in recent if e.get("outcome") == "L")
    pending = sum(1 for e in recent if e.get("outcome") is None)
    decided = wins + losses

    by_market: dict[str, dict] = {}
    for e in recent:
        m = e.get("market", "unknown")
        if m not in by_market:
            by_market[m] = {"total": 0, "wins": 0, "losses": 0, "pending": 0}
        bm = by_market[m]
        bm["total"] += 1
        if e.get("outcome") == "W":
            bm["wins"] += 1
        elif e.get("outcome") == "L":
            bm["losses"] += 1
        else:
            bm["pending"] += 1

    return {
        "total":    total,
        "wins":     wins,
        "losses":   losses,
        "pending":  pending,
        "win_rate": wins / decided if decided else None,
        "by_market": by_market,
        "entries":  sorted(recent, key=lambda x: x["date"], reverse=True),
    }
