"""Free MLB odds scraper using Bovada's public JSON endpoint.

Bovada exposes a documented (in the sense of "extensively reverse-engineered
in many open-source repos") JSON feed with no authentication. We fetch the
MLB coupon, walk the displayGroups, and normalize markets into the same
shape our value engine expects.

What we extract:
  - Game lines: moneyline, total runs, run line (-1.5/+1.5)
  - Pitcher props: total strikeouts (over/under with line)
  - Batter props: HR, 2+ HR, Hit, 2+ Hits, Run, RBI, Total Bases
                  (Bovada quotes these as Yes/No on a threshold; we convert
                   them to over/under with a half-point line)

Caveats / TOS:
  - Bovada's TOS prohibits automated scraping. Personal/educational use is
    very common but you should respect rate limits and not hammer them.
  - Their pricing skews wider than US books (more juice baked in), which
    actually makes finding +EV harder, not easier — model-vs-Bovada edges
    that survive their juice are usually real.
  - Use polite caching (30-60s TTL is fine; lines move slowly pre-game).
"""
from __future__ import annotations
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import requests

CACHE = Path(__file__).resolve().parent.parent / "data" / "odds"
CACHE.mkdir(parents=True, exist_ok=True)

URL = "https://www.bovada.lv/services/sports/event/coupon/events/A/description/baseball/mlb"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# Map a Bovada market description to a (market_key, line_inferred) pair.
# Lines for batter props are inferred from the market name (e.g. "2+ Hits" -> 1.5).
BATTER_PROP_MAP = [
    # (regex, market_key, inferred_over_line)
    (re.compile(r"^Player to hit a Home Run$", re.I), "hr", 0.5),
    (re.compile(r"^Player to hit 2\+ Home Runs$", re.I), "hr", 1.5),
    (re.compile(r"^Player to record a Hit$", re.I), "hits", 0.5),
    (re.compile(r"^Player to record 2\+ Hits$", re.I), "hits", 1.5),
    (re.compile(r"^Player to record 3\+ Hits$", re.I), "hits", 2.5),
    (re.compile(r"^Player to record a Run$", re.I), "runs", 0.5),
    (re.compile(r"^Player to record 2\+ Runs$", re.I), "runs", 1.5),
    (re.compile(r"^Player to record an RBI$", re.I), "rbi", 0.5),
    (re.compile(r"^Player to record 2\+ RBIs?$", re.I), "rbi", 1.5),
    (re.compile(r"^Player to record 3\+ RBIs?$", re.I), "rbi", 2.5),
    (re.compile(r"^Player to record (\d+)\+ Total Bases$", re.I), "tb", None),
    (re.compile(r"^Player to record a Stolen Base$", re.I), "sb", 0.5),
]

PITCHER_PROP_MAP = [
    (re.compile(r"^Total Strikeouts - .+$", re.I), "pitcher_k"),
    (re.compile(r"^Will (.+) Record a Win\?$", re.I), "pitcher_win"),
]


def _fetch(force: bool = False, ttl_seconds: int = 60) -> list[dict]:
    cf = CACHE / "bovada_mlb.json"
    if not force and cf.exists() and (time.time() - cf.stat().st_mtime) < ttl_seconds:
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    r = requests.get(URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    cf.write_text(json.dumps(data), encoding="utf-8")
    return data


def _extract_player_name(outcome_desc: str) -> str:
    """Bovada outcomes look like 'Junior Caminero (TB)' — strip the team tag."""
    return re.sub(r"\s*\([A-Z]{2,4}\)\s*$", "", outcome_desc).strip()


def _amer(outcome: dict) -> Optional[int]:
    raw = (outcome.get("price") or {}).get("american")
    if raw in (None, "", "EVEN", "even"):
        return 100 if raw and raw.upper() == "EVEN" else None
    try:
        return int(str(raw).replace("+", "").replace(" ", ""))
    except (ValueError, TypeError):
        return None


def parse_mlb_lines() -> dict:
    """Return a normalized dict:
        {
          "as_of": iso-timestamp,
          "games": [{home_team, away_team, commence_time, moneyline, total, run_line}, ...],
          "player_props": [{player, market, line, over, under}, ...],
        }
    """
    raw = _fetch()
    if not raw or not isinstance(raw, list) or not raw[0].get("events"):
        return {"as_of": datetime.now(timezone.utc).isoformat(), "games": [], "player_props": []}

    out_games: list[dict] = []
    out_props: list[dict] = []

    for event in raw[0]["events"]:
        desc = event.get("description", "")
        if " @ " not in desc:
            continue
        away, home = [s.strip() for s in desc.split(" @ ", 1)]
        start_ms = event.get("startTime")
        commence = (datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
                    if start_ms else None)

        ml: dict | None = None
        total: dict | None = None
        run_line: dict | None = None

        for dg in event.get("displayGroups", []):
            grp = dg.get("description", "")
            for m in dg.get("markets", []):
                mdesc = m.get("description", "")
                period = (m.get("period") or {}).get("description", "")
                if period and period not in ("Match", "Game", "Live", "Regulation Time"):
                    continue
                outs = m.get("outcomes", [])
                if grp == "Game Lines":
                    if mdesc == "Moneyline" and len(outs) == 2:
                        h_p = next((_amer(o) for o in outs if o["description"] == home), None)
                        a_p = next((_amer(o) for o in outs if o["description"] == away), None)
                        if h_p is not None and a_p is not None:
                            ml = {"home": h_p, "away": a_p}
                    elif mdesc == "Total" and len(outs) == 2:
                        ov = next((o for o in outs if o["description"] == "Over"), None)
                        un = next((o for o in outs if o["description"] == "Under"), None)
                        if ov and un:
                            line = (ov.get("price") or {}).get("handicap")
                            try:
                                line_f = float(line)
                            except (TypeError, ValueError):
                                line_f = None
                            if line_f is not None:
                                total = {"line": line_f, "over": _amer(ov), "under": _amer(un)}
                    elif mdesc == "Runline" and len(outs) == 2:
                        h_o = next((o for o in outs if o["description"] == home), None)
                        a_o = next((o for o in outs if o["description"] == away), None)
                        if h_o and a_o:
                            try:
                                hp = (h_o.get("price") or {}).get("handicap")
                                line = abs(float(hp)) if hp is not None else 1.5
                            except (TypeError, ValueError):
                                line = 1.5
                            # Bovada gives home's handicap; we encode as
                            # negative for home favorite (home -1.5).
                            # The favorite is whoever has the negative line.
                            try:
                                home_handi = float((h_o.get("price") or {}).get("handicap", 0))
                            except (TypeError, ValueError):
                                home_handi = 0.0
                            run_line = {
                                "line": -1.5 if home_handi < 0 else 1.5,
                                "home": _amer(h_o), "away": _amer(a_o),
                            }
                elif grp == "Pitcher Props":
                    # "Total Strikeouts - Nick Martinez (TB)" with handicap line
                    name_match = re.match(r"^Total Strikeouts - (.+) \([A-Z]{2,4}\)$", mdesc)
                    if name_match and len(outs) == 2:
                        name = name_match.group(1).strip()
                        ov = next((o for o in outs if o["description"] == "Over"), None)
                        un = next((o for o in outs if o["description"] == "Under"), None)
                        if ov and un:
                            try:
                                line_f = float((ov.get("price") or {}).get("handicap"))
                            except (TypeError, ValueError):
                                line_f = None
                            if line_f is not None:
                                out_props.append({
                                    "game": desc, "player": name, "market": "pitcher_k",
                                    "line": line_f,
                                    "over": _amer(ov), "under": _amer(un),
                                    "source": "bovada",
                                })
                elif grp == "Player Props":
                    # Boolean-style props: "Player to hit a Home Run" with N outcomes
                    # of (PlayerName (TEAM), price)
                    market_key, inferred_line = None, None
                    for rx, key, line_val in BATTER_PROP_MAP:
                        m_ = rx.match(mdesc)
                        if m_:
                            market_key = key
                            if line_val is None and key == "tb":
                                # "X+ Total Bases" -> line = X - 0.5
                                inferred_line = float(m_.group(1)) - 0.5
                            else:
                                inferred_line = line_val
                            break
                    if market_key is None:
                        continue
                    # Each outcome is a player; assume "Yes" priced. Bovada
                    # sometimes pairs Yes/No; we take the listed price as Yes.
                    for o in outs:
                        name = _extract_player_name(o.get("description", ""))
                        am = _amer(o)
                        if not name or am is None:
                            continue
                        # No-side price is implied — we mark UNDER as opposite
                        # implied prob with 0 vig adjustment (rough; we'll
                        # devig in value.py using only the OVER side).
                        out_props.append({
                            "game": desc, "player": name, "market": market_key,
                            "line": inferred_line,
                            "over": am, "under": None,
                            "source": "bovada",
                        })

        out_games.append({
            "home_team": home, "away_team": away,
            "commence_time": commence,
            "moneyline": ml, "total": total, "run_line": run_line,
        })

    return {"as_of": datetime.now(timezone.utc).isoformat(),
            "games": out_games, "player_props": out_props}


def fetch_consensus(force: bool = False) -> dict:
    """Public entrypoint matching the shape used by `odds.consensus_lines`."""
    return parse_mlb_lines()
