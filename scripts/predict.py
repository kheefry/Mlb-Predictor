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
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict_core import predict_slate


def _fmt_amer(o: int) -> str:
    return f"+{o}" if o > 0 else str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_date", nargs="?", default=None,
                    help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--top-bets", type=int, default=15,
                    help="show top-N value bets across all games")
    ap.add_argument("--edge", type=float, default=0.03,
                    help="minimum edge to flag (default 0.03)")
    ap.add_argument("--no-odds", action="store_true",
                    help="skip live odds lookup")
    args = ap.parse_args()

    target = (datetime.fromisoformat(args.game_date).date()
              if args.game_date else datetime.now(timezone.utc).date())

    print(f"\n=== MLB predictions for {target} ===")

    result = predict_slate(
        target_date=target,
        edge_threshold=args.edge,
        fetch_odds=not args.no_odds,
        top_n=args.top_bets,
    )

    if not result.games:
        print("No games scheduled.")
        return

    print(f"Found {result.n_games} games.  Odds: {result.odds_source}  "
          f"Props loaded: {result.n_props_loaded}\n")

    for gp in result.games:
        print(f"--- {gp.away_team} @ {gp.home_team} ---")
        print(f"  Venue: {gp.venue}  [{gp.park_roof}]   "
              f"Temp {gp.temp_f:.0f}F  Wind to CF {gp.wind_to_cf_mph:+.1f} mph  "
              f"runs_mult {gp.runs_mult:.2f}  hr_mult {gp.hr_mult:.2f}")
        print(f"  SP: {gp.away_sp_name or '?'} (FIP {gp.away_sp_fip:.2f})  vs  "
              f"{gp.home_sp_name or '?'} (FIP {gp.home_sp_fip:.2f})"
              + ("" if gp.starters_confirmed else "  [UNCONFIRMED]"))
        print(f"  Predicted: {gp.away_team[:16]:16s} {gp.pred_away_runs:.2f}  -  "
              f"{gp.pred_home_runs:.2f}  {gp.home_team}")
        print(f"  Total: {gp.pred_total:.2f}    P(home win): {gp.p_home_win:.1%}    "
              f"P(over 8.5): {gp.p_over_8_5:.1%}")

        for label, batters in [
            (gp.away_team, gp.away_batters),
            (gp.home_team, gp.home_batters),
        ]:
            if not batters:
                continue
            print(f"\n  {label} batters:")
            print(f"    {'Player':24s} PA   H   HR   TB   RBI  R    K    BB")
            for b in batters:
                print(f"    {b['name'][:24]:24s} {b['expected_pa']:.1f}  {b['proj_h']:.2f} "
                      f"{b['proj_hr']:.2f}  {b['proj_tb']:.2f} {b['proj_rbi']:.2f} "
                      f"{b['proj_runs']:.2f} {b['proj_k']:.2f} {b['proj_bb']:.2f}")

        for label, starter in [
            (gp.away_team, gp.away_starter),
            (gp.home_team, gp.home_starter),
        ]:
            if not starter:
                continue
            print(f"  {label} starter ({starter['name']}): "
                  f"{starter['expected_ip']:.1f} IP  {starter['proj_k']:.1f} K  "
                  f"{starter['proj_bb']:.1f} BB  {starter['proj_h']:.1f} H  "
                  f"{starter['proj_er']:.1f} ER  {starter['proj_hr_allowed']:.2f} HR")

        if gp.book:
            ml  = (gp.book.get("moneyline") or {})
            tot = (gp.book.get("total")     or {})
            rl  = (gp.book.get("run_line")  or {})
            print(f"\n  Book ({gp.book_source}): ML {gp.away_team[:8]} "
                  f"{_fmt_amer(ml.get('away', 0))} / {gp.home_team[:8]} "
                  f"{_fmt_amer(ml.get('home', 0))}   "
                  f"Total {tot.get('line', '?')}   RL +/-{abs(rl.get('line', 1.5)):.1f}")

        if gp.game_value:
            print("  Game value:")
            for vb in sorted(gp.game_value, key=lambda x: -x["edge_pct"]):
                print(f"    {vb['description']:40s} {_fmt_amer(vb['odds']):>5s}  "
                      f"model {vb['model_prob']:.1%}  edge +{vb['edge_pct']:.1f}%  "
                      f"Kelly {vb['kelly']:.2%}")

        if gp.prop_value:
            print("  Player prop value:")
            for vb in sorted(gp.prop_value, key=lambda x: -x["edge_pct"]):
                print(f"    {vb['description']:45s} {_fmt_amer(vb['odds']):>5s}  "
                      f"model {vb['model_prob']:.1%}  edge +{vb['edge_pct']:.1f}%  Kelly {vb['kelly']:.2%}")

        print()

    if result.concentration_warning:
        print(f"[CONCENTRATION WARNING] {result.concentration_warning}\n")

    if result.top_value:
        print("=" * 88)
        print(f"TOP {len(result.top_value)} VALUE BETS "
              f"(ranked by Score = edge x stat-reliability x outcome-info)")
        print("=" * 88)
        print(f"{'Bet':<48s} {'Odds':>6s}  {'Model':>7s}  {'Edge':>6s}  "
              f"{'Score':>6s}  {'EV/$':>7s}  {'Kelly':>6s}")
        for vb in result.top_value:
            print(f"{vb['description'][:48]:<48s} {_fmt_amer(vb['odds']):>6s}  "
                  f"{vb['model_prob']:>6.1%}  +{vb['edge_pct']:>4.1f}%  "
                  f"{vb.get('score', 0):>6.2f}  "
                  f"{vb['ev_per_dollar']:>+.3f}  {vb['kelly']:>5.2%}")


if __name__ == "__main__":
    main()
