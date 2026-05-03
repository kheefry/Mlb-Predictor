"""Predict today's slate end-to-end.

Run:
    python -m scripts.predict                # today
    python -m scripts.predict 2026-04-28     # specific date
    python -m scripts.predict --top-bets 10  # show top N value bets

Outputs (per game):
  - Team score predictions (with weather forecast for first pitch)
  - Win probability, total runs forecast
  - Top batter projections
  - Starter projections
  - Value bets vs sportsbook lines (if available)
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import mlb_api, parks, weather, features as feats
from src import model as mdl, projections as proj, odds, value, name_match


def _fmt_amer(o: int) -> str:
    return f"+{o}" if o > 0 else str(o)


def _stats_lookup(d: dict) -> dict[int, dict]:
    return {int(k): v for k, v in d.items()}


def _team_name_match(needle: str, haystack: str) -> bool:
    n = needle.lower(); h = haystack.lower()
    return n in h or h in n


def _find_book(books: list[dict], home: str, away: str) -> dict | None:
    for b in books:
        if _team_name_match(b["home_team"], home) and _team_name_match(b["away_team"], away):
            return b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_date", nargs="?", default=None,
                    help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--top-bets", type=int, default=15,
                    help="show top-N value bets across all games")
    ap.add_argument("--edge", type=float, default=0.03,
                    help="minimum edge (model_p - novig_p) to flag (default 0.03)")
    ap.add_argument("--no-odds", action="store_true",
                    help="skip live odds lookup")
    args = ap.parse_args()

    target = (datetime.fromisoformat(args.game_date).date()
              if args.game_date else datetime.now(timezone.utc).date())

    print(f"\n=== MLB predictions for {target} ===")

    # --- Load model + season-stat snapshot ---
    snap = json.loads((ROOT / "data" / "games" / "snapshot_2026.json").read_text(encoding="utf-8"))
    team_off = _stats_lookup(snap["team_off"])
    team_pit = _stats_lookup(snap["team_pit"])
    pitcher_stats = _stats_lookup(snap["pitcher_stats"])
    batter_stats = _stats_lookup(snap["batter_stats"])
    bat_recent = _stats_lookup(snap.get("batter_stats_recent", {}))
    pit_recent = _stats_lookup(snap.get("pitcher_stats_recent", {}))
    bat_vs_l = _stats_lookup(snap.get("bat_vs_l", {}))
    bat_vs_r = _stats_lookup(snap.get("bat_vs_r", {}))
    bat_sides = {int(k): v for k, v in snap.get("bat_sides", {}).items()}
    pit_throws = {int(k): v for k, v in snap.get("pit_throws", {}).items()}

    model = mdl.TeamScoreModel.load(ROOT / "data" / "models" / "team_runs.joblib")

    # --- Pull today's games ---
    games = mlb_api.schedule(target)
    if not games:
        print("No games scheduled.")
        return

    print(f"Found {len(games)} games.\n")

    # --- Live odds (if available) ---
    book_data: list[dict] = []
    live_props: list[dict] = []
    odds_source = "none"
    manual = odds.load_manual()
    if not args.no_odds:
        try:
            book_data, live_props, odds_source = odds.load_lines_with_fallback()
            if book_data or live_props:
                print(f"Loaded {len(book_data)} game line(s) and {len(live_props)} "
                      f"player prop(s) from: {odds_source}\n")
            else:
                print("(No live odds available — falling through to manual file if present)\n")
        except Exception as e:
            print(f"(Odds fetch failed: {e})")
    # Merge: live_props takes precedence; manual props extend it
    merged_props = list(live_props)
    if manual.get("player_props"):
        merged_props.extend(manual["player_props"])

    all_value_bets: list[value.ValueBet] = []

    for g in games:
        st = (g.get("status") or {}).get("codedGameState", "")
        if st in ("D", "C"):
            continue

        f = feats.build_game_features(g, team_off, team_pit, pitcher_stats)
        if f is None:
            continue

        lineups = mlb_api.extract_lineups(g)
        home_lineup_ids = lineups["home"] or None
        away_lineup_ids = lineups["away"] or None

        # Predict team runs
        long = mdl.long_form(pd.DataFrame([asdict(f)]))
        long["pred_runs"] = model.predict_runs(long)
        home_pred = float(long[long["is_home"] == 1].iloc[0]["pred_runs"])
        away_pred = float(long[long["is_home"] == 0].iloc[0]["pred_runs"])

        p_home_win = value.home_win_prob(home_pred, away_pred)
        total_pred = home_pred + away_pred
        p_over_8_5 = value.total_over_prob(home_pred, away_pred, 8.5)

        # Local time of first pitch (rough — using park lon to estimate offset)
        park = parks.get_park(f.venue)
        utc_dt = mlb_api.parse_game_time(g)

        roof_note = f" [{park.roof}]" if park.roof != "open" else ""
        print(f"--- {f.away_team} @ {f.home_team} ---")
        print(f"  Venue: {f.venue}{roof_note}   First pitch: {utc_dt.strftime('%H:%M UTC')}")
        print(f"  Weather (game time): {f.temp_f:.0f}F, wind to CF {f.wind_to_cf_mph:+.1f} mph"
              f", runs_mult {f.runs_mult:.2f}, hr_mult {f.hr_mult:.2f}")
        print(f"  SP: {f.away_sp_name or '?'} (FIP {f.away_sp_fip:.2f})  vs  "
              f"{f.home_sp_name or '?'} (FIP {f.home_sp_fip:.2f})")
        print(f"  Predicted score: {f.away_team[:18]:18s} {away_pred:.2f}  -  "
              f"{home_pred:.2f}  {f.home_team}")
        print(f"  Predicted total: {total_pred:.2f}     P(home win): {p_home_win:.1%}     "
              f"P(over 8.5): {p_over_8_5:.1%}")

        # Player projections — use season-PA leaderboard for likely lineup
        for side, tid, otid, sp_id_self, sp_id_opp, team_pred, opp_pred, team_label, lineup_ids in [
            ("away", f.away_team_id, f.home_team_id, f.away_sp_id, f.home_sp_id, away_pred, home_pred, f.away_team, away_lineup_ids),
            ("home", f.home_team_id, f.away_team_id, f.home_sp_id, f.away_sp_id, home_pred, away_pred, f.home_team, home_lineup_ids),
        ]:
            opp_sp_q = feats.pitcher_quality_index(pitcher_stats.get(sp_id_opp, {})) if sp_id_opp else feats.pitcher_quality_index({})
            opp_off_idx = feats.team_offense_index(team_off.get(otid, {}))
            wadj = {"runs_mult": f.runs_mult, "hr_mult": f.hr_mult,
                    "wind_to_cf_mph": f.wind_to_cf_mph, "temp_f": f.temp_f}

            batters = proj.get_likely_batters(tid, batter_stats, lineup_ids=lineup_ids)
            print(f"\n  {team_label} batters (top of likely order):")
            print(f"    {'Player':24s} PA   H   HR   TB   RBI  R    K    BB")
            order = 1
            batter_projs = []
            for bs in batters:
                pid = int(bs.get("player_id") or 0)
                pl = proj.resolve_platoon(pid, sp_id_opp, bat_sides, pit_throws,
                                          bat_vs_l, bat_vs_r)
                rs = bat_recent.get(pid)
                p = proj.project_batter(bs, order, team_pred, opp_sp_q, park, wadj,
                                        recent_stats=rs,
                                        bat_side=pl["bat_side"],
                                        opp_pit_throws=pl["opp_pit_throws"],
                                        bat_split=pl["bat_split"],
                                        is_switch=pl.get("is_switch", False))
                batter_projs.append(p)
                print(f"    {p.name[:24]:24s} {p.expected_pa:.1f}  {p.proj_h:.2f} {p.proj_hr:.2f}"
                      f"  {p.proj_tb:.2f} {p.proj_rbi:.2f} {p.proj_runs:.2f} {p.proj_k:.2f} {p.proj_bb:.2f}")
                order += 1

            # Pitcher projection (probable starter)
            if sp_id_self:
                ps = pitcher_stats.get(sp_id_self, {"player_id": sp_id_self, "name": "?", "team_id": tid})
                ps_recent = pit_recent.get(int(sp_id_self))
                pp = proj.project_pitcher(ps, tid, opp_off_idx, opp_pred, park, wadj,
                                          recent_stats=ps_recent)
                print(f"  {team_label} starter ({pp.name}): "
                      f"{pp.expected_ip:.1f} IP, {pp.proj_k:.1f} K, {pp.proj_bb:.1f} BB, "
                      f"{pp.proj_h:.1f} H, {pp.proj_er:.1f} ER, {pp.proj_hr_allowed:.2f} HR allowed")

        # ---- Value bets ----
        bk = _find_book(book_data, f.home_team, f.away_team) if book_data else None
        if bk is None and manual:
            for mg in manual.get("games", []):
                if (_team_name_match(mg.get("home_team", ""), f.home_team) and
                    _team_name_match(mg.get("away_team", ""), f.away_team)):
                    bk = mg
                    break

        if bk:
            ml = bk.get("moneyline") or {}
            tot = bk.get("total") or {}
            rl = bk.get("run_line") or {}
            print(f"\n  Sportsbook lines: ML {f.away_team[:8]} {_fmt_amer(ml.get('away', 0))} "
                  f"/ {f.home_team[:8]} {_fmt_amer(ml.get('home', 0))}   "
                  f"Total {tot.get('line', '?')}   "
                  f"RL +/-{abs(rl.get('line', 1.5))}")
            game_value = value.evaluate_game_lines(
                f.home_team, f.away_team, home_pred, away_pred, bk,
                edge_threshold=args.edge,
            )
            if game_value:
                print("  Value found:")
                for vb in sorted(game_value, key=lambda x: -x.edge_pct):
                    print(f"    {vb.description:35s} {_fmt_amer(vb.odds):>5s}  "
                          f"model {vb.model_prob:.1%} vs no-vig {vb.novig_prob:.1%}  "
                          f"edge +{vb.edge_pct:.1f}%  EV {vb.ev_per_dollar:+.3f}  Kelly {vb.kelly:.2%}")
                all_value_bets.extend(game_value)
        else:
            if not args.no_odds:
                print("  (no book line found for this game)")

        # ---- Player props (live + manual merged) ----
        if merged_props:
            # Build a quick name lookup of all batter projections we made for this game
            # (re-project to be safe for any name in props)
            by_name = {}
            for bs in (proj.get_likely_batters(f.home_team_id, batter_stats, lineup_ids=home_lineup_ids) +
                       proj.get_likely_batters(f.away_team_id, batter_stats, lineup_ids=away_lineup_ids)):
                by_name[bs.get("name", "")] = bs
            # Pitchers
            for sp_id in (f.home_sp_id, f.away_sp_id):
                if sp_id and pitcher_stats.get(sp_id):
                    by_name[pitcher_stats[sp_id].get("name", "")] = pitcher_stats[sp_id]
            # Filter to props for *this* game (Bovada gives a "game" field; manual may not)
            our_props = [pp for pp in merged_props
                         if "game" not in pp
                         or (_team_name_match(pp["game"].split(" @ ")[0], f.away_team)
                             and _team_name_match(pp["game"].split(" @ ")[1], f.home_team))]

            game_prop_value = []
            for pp in our_props:
                name = pp.get("player", "")
                # Robust fuzzy match (handles accents, Jr., etc.)
                resolved = name_match.find_match(name, by_name.keys())
                pdata = by_name.get(resolved) if resolved else None
                if not pdata:
                    continue
                # Determine team / matchup for projection
                is_pitcher = pp["market"].startswith("pitcher_") or pp["market"] in ("outs", "ip")
                if is_pitcher:
                    is_home_pitcher = (pdata.get("player_id") == f.home_sp_id)
                    team_id = f.home_team_id if is_home_pitcher else f.away_team_id
                    opp_id = f.away_team_id if is_home_pitcher else f.home_team_id
                    opp_off = feats.team_offense_index(team_off.get(opp_id, {}))
                    opp_pred = away_pred if is_home_pitcher else home_pred
                    pproj = proj.project_pitcher(pdata, team_id, opp_off, opp_pred, park,
                                                 {"runs_mult": f.runs_mult, "hr_mult": f.hr_mult})
                    means = {
                        "pitcher_k": pproj.proj_k, "pitcher_outs": pproj.expected_outs,
                        "pitcher_er": pproj.proj_er, "pitcher_h": pproj.proj_h,
                        "pitcher_bb": pproj.proj_bb, "pitcher_hr": pproj.proj_hr_allowed,
                    }
                    mean = means.get(pp["market"])
                else:
                    is_home_batter = (pdata.get("team_id") == f.home_team_id)
                    sp_id = f.away_sp_id if is_home_batter else f.home_sp_id
                    opp_sp_q = feats.pitcher_quality_index(pitcher_stats.get(sp_id, {})) if sp_id else feats.pitcher_quality_index({})
                    team_pred_local = home_pred if is_home_batter else away_pred
                    bproj = proj.project_batter(pdata, 3, team_pred_local, opp_sp_q, park,
                                                {"runs_mult": f.runs_mult, "hr_mult": f.hr_mult,
                                                 "wind_to_cf_mph": f.wind_to_cf_mph, "temp_f": f.temp_f})
                    means = {
                        "hr": bproj.proj_hr, "hits": bproj.proj_h, "tb": bproj.proj_tb,
                        "rbi": bproj.proj_rbi, "runs": bproj.proj_runs, "k": bproj.proj_k,
                        "bb": bproj.proj_bb, "sb": bproj.proj_sb,
                    }
                    mean = means.get(pp["market"])
                if mean is None:
                    continue
                vbs = value.evaluate_prop(name, pp["market"], mean, pp["line"],
                                          pp.get("over"), pp.get("under"),
                                          edge_threshold=args.edge)
                if vbs:
                    game_prop_value.extend(vbs)
                    all_value_bets.extend(vbs)
            if game_prop_value:
                print("  Player prop value:")
                for vb in sorted(game_prop_value, key=lambda x: -x.edge_pct):
                    print(f"    {vb.description:45s} {_fmt_amer(vb.odds):>5s}  "
                          f"model {vb.model_prob:.1%}  edge +{vb.edge_pct:.1f}%  Kelly {vb.kelly:.2%}")

        print()

    # ---- Slate-wide leaderboard ----
    if all_value_bets:
        print("=" * 88)
        print(f"TOP {min(args.top_bets, len(all_value_bets))} VALUE BETS "
              f"(ranked by Score = edge × stat-reliability × outcome-info)")
        print("=" * 88)
        ranked = sorted(all_value_bets,
                        key=lambda x: -getattr(x, "score", x.edge_pct))[:args.top_bets]
        print(f"{'Bet':<48s} {'Odds':>6s}  {'Model':>7s}  {'Edge':>6s}  {'Score':>6s}  {'EV/$':>7s}  {'Kelly':>6s}")
        for vb in ranked:
            print(f"{vb.description[:48]:<48s} {_fmt_amer(vb.odds):>6s}  "
                  f"{vb.model_prob:>6.1%}  +{vb.edge_pct:>4.1f}%  "
                  f"{getattr(vb, 'score', 0):>6.2f}  "
                  f"{vb.ev_per_dollar:>+.3f}  {vb.kelly:>5.2%}")


if __name__ == "__main__":
    main()
