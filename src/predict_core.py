"""Core prediction logic — used by both the CLI and the Streamlit app.

`predict_slate()` returns a structured `SlateResult` containing per-game
predictions, projections, sportsbook lines, and value bets. Pure data — no
printing or plotting.
"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import mlb_api, parks, weather, features as feats, statcast as sc
from . import model as mdl, projections as proj, odds, value, name_match, bet_tracker, umpire as ump


ROOT = Path(__file__).resolve().parent.parent


# ---------- Data classes ----------
@dataclass
class GamePrediction:
    game_pk: int
    date: str
    venue: str
    park_roof: str
    park_pf_runs: float
    park_pf_hr: float

    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    first_pitch_utc: str             # ISO timestamp

    home_sp_id: Optional[int]
    home_sp_name: str
    home_sp_fip: float
    home_sp_xfip: float
    away_sp_id: Optional[int]
    away_sp_name: str
    away_sp_fip: float
    away_sp_xfip: float

    # Weather for first pitch
    temp_f: float
    wind_to_cf_mph: float
    runs_mult: float
    hr_mult: float

    # Predictions
    pred_home_runs: float
    pred_away_runs: float
    pred_total: float
    p_home_win: float
    p_over_8_5: float

    # Player projections
    home_batters: list[dict] = field(default_factory=list)
    away_batters: list[dict] = field(default_factory=list)
    home_starter: Optional[dict] = None
    away_starter: Optional[dict] = None

    # Sportsbook lines (None if not available)
    book: Optional[dict] = None
    book_source: str = "none"

    # Value bets (filtered to >= edge_threshold)
    game_value: list[dict] = field(default_factory=list)
    prop_value: list[dict] = field(default_factory=list)
    # Every evaluated bet for this game, regardless of edge — used by the
    # Pure Confidence leaderboard (model-certainty only, ignores book agreement).
    all_bets: list[dict] = field(default_factory=list)


@dataclass
class SlateResult:
    target_date: str
    odds_source: str
    n_games: int
    n_books: int
    n_props_loaded: int
    games: list[GamePrediction]
    top_value: list[dict]            # ranked across slate
    concentration_warning: Optional[str] = None
    # Slate-wide unfiltered bet pool for Pure Confidence ranking.
    all_bets: list[dict] = field(default_factory=list)


# ---------- Helpers ----------
def _stats_lookup(d: dict) -> dict[int, dict]:
    return {int(k): v for k, v in d.items()}


def _team_name_match(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    n = needle.lower(); h = haystack.lower()
    return n in h or h in n


def _find_book(books: list[dict], home: str, away: str) -> dict | None:
    for b in books:
        if _team_name_match(b.get("home_team", ""), home) and _team_name_match(b.get("away_team", ""), away):
            return b
    return None


def _vb_to_dict(vb: value.ValueBet) -> dict:
    """Render a ValueBet as a JSON-friendly dict."""
    d = asdict(vb)
    d["edge_pct"] = round(d["edge_pct"], 2)
    d["model_prob"] = round(d["model_prob"], 4)
    d["novig_prob"] = round(d["novig_prob"], 4)
    d["ev_per_dollar"] = round(d["ev_per_dollar"], 4)
    d["kelly"] = round(d["kelly"], 4)
    d["confidence"] = round(d.get("confidence", 0.0), 4)
    d["score"] = round(d.get("score", 0.0), 3)
    return d


# ---------- Main entry point ----------
def predict_slate(target_date: date | str | None = None,
                  edge_threshold: float = 0.03,
                  fetch_odds: bool = True,
                  top_n: int = 30) -> SlateResult:
    """Predict the slate for `target_date` (default = today UTC).

    Returns a SlateResult that's safe to pass to UI code or serialize to JSON.
    """
    if target_date is None:
        target = datetime.now(timezone.utc).date()
    elif isinstance(target_date, str):
        target = datetime.fromisoformat(target_date).date()
    else:
        target = target_date

    # Statcast — fetched after snap so we can use bat_stats for the player→team map
    # (deferred below, after snap is loaded)

    # Load model + season-stat snapshot
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

    # Statcast: build player→team map from MLB API bat_stats, then fetch
    try:
        _player_team_map = {pid: int(s["team_id"])
                            for pid, s in batter_stats.items() if s.get("team_id")}
        sc_team_bat = sc.get_team_batting(2026, _player_team_map)
        sc_pit_data = sc.get_pitcher_stats(2026)
        sc_bat_data = sc.get_batter_stats(2026)
    except Exception:
        sc_team_bat, sc_pit_data, sc_bat_data = {}, {}, {}

    model = mdl.TeamScoreModel.load(ROOT / "data" / "models" / "team_runs.joblib")
    _model_dir = ROOT / "data" / "models"
    _boot = mdl.load_bootstrap_ensemble(_model_dir)
    _temporal = mdl.load_temporal_ensemble(_model_dir)
    _all_models = [model] + _boot + _temporal   # main + bootstrap + temporal
    _ump_rates = ump.load_rates()

    # Pull schedule
    games_raw = mlb_api.schedule(target)
    if not games_raw:
        return SlateResult(target.isoformat(), "none", 0, 0, 0, [], [])

    # Live odds (skip cleanly if unavailable)
    book_data: list[dict] = []
    live_props: list[dict] = []
    odds_source = "none"
    if fetch_odds:
        try:
            book_data, live_props, odds_source = odds.load_lines_with_fallback()
            if book_data:
                odds.snapshot_odds(book_data, live_props)
        except Exception:
            book_data, live_props, odds_source = [], [], "error"
    manual = odds.load_manual()
    merged_props = list(live_props)
    if manual.get("player_props"):
        merged_props.extend(manual["player_props"])

    all_value_bets: list[value.ValueBet] = []
    games_out: list[GamePrediction] = []

    for g in games_raw:
        st = (g.get("status") or {}).get("codedGameState", "")
        if st in ("D", "C"):
            continue

        # Pull confirmed lineups first so they feed into the team-runs features.
        lineups = mlb_api.extract_lineups(g)
        home_lineup_ids = lineups["home"] or None
        away_lineup_ids = lineups["away"] or None

        f = feats.build_game_features(g, team_off, team_pit, pitcher_stats,
                                      sc_team_bat=sc_team_bat, sc_pit=sc_pit_data,
                                      home_lineup_ids=home_lineup_ids,
                                      away_lineup_ids=away_lineup_ids,
                                      batter_stats=batter_stats,
                                      bat_vs_l=bat_vs_l, bat_vs_r=bat_vs_r,
                                      bat_sides=bat_sides, pit_throws=pit_throws)
        if f is None:
            continue

        # Umpire: try game feed (2-hr cache) for today's games; fall back to 1.0
        try:
            _feed = mlb_api._get(f"/v1.1/game/{f.game_pk}/feed/live", ttl_seconds=7200)
            _hp = ump.get_hp_umpire_from_game_feed(_feed)
            f.ump_k_mult = ump.get_k_mult(_hp, _ump_rates)
        except Exception:
            f.ump_k_mult = 1.0

        long = mdl.long_form(pd.DataFrame([asdict(f)]))
        long["pred_runs"] = mdl.predict_ensemble(_all_models, long)
        home_pred = float(long[long["is_home"] == 1].iloc[0]["pred_runs"])
        away_pred = float(long[long["is_home"] == 0].iloc[0]["pred_runs"])

        p_home_win = value.home_win_prob(home_pred, away_pred)
        p_over_8_5 = value.total_over_prob(home_pred, away_pred, 8.5)
        total_pred = home_pred + away_pred

        park = parks.get_park(f.venue)
        utc_dt = mlb_api.parse_game_time(g)

        gp = GamePrediction(
            game_pk=int(f.game_pk), date=f.date, venue=f.venue,
            park_roof=park.roof, park_pf_runs=park.pf_runs, park_pf_hr=park.pf_hr,
            home_team=f.home_team, away_team=f.away_team,
            home_team_id=f.home_team_id, away_team_id=f.away_team_id,
            first_pitch_utc=utc_dt.isoformat(),

            home_sp_id=f.home_sp_id, home_sp_name=f.home_sp_name,
            home_sp_fip=f.home_sp_fip, home_sp_xfip=f.home_sp_xfip,
            away_sp_id=f.away_sp_id, away_sp_name=f.away_sp_name,
            away_sp_fip=f.away_sp_fip, away_sp_xfip=f.away_sp_xfip,

            temp_f=f.temp_f, wind_to_cf_mph=f.wind_to_cf_mph,
            runs_mult=f.runs_mult, hr_mult=f.hr_mult,

            pred_home_runs=home_pred, pred_away_runs=away_pred,
            pred_total=total_pred, p_home_win=p_home_win, p_over_8_5=p_over_8_5,
        )

        # Build player projections
        for side, tid, otid, sp_id_self, sp_id_opp, team_pred, opp_pred, lineup_ids, opp_lineup_ids in [
            ("away", f.away_team_id, f.home_team_id, f.away_sp_id, f.home_sp_id, away_pred, home_pred, away_lineup_ids, home_lineup_ids),
            ("home", f.home_team_id, f.away_team_id, f.home_sp_id, f.away_sp_id, home_pred, away_pred, home_lineup_ids, away_lineup_ids),
        ]:
            opp_sp_q = feats.pitcher_quality_index(pitcher_stats.get(sp_id_opp, {})) if sp_id_opp else feats.pitcher_quality_index({})
            opp_off_idx = feats.team_offense_index(team_off.get(otid, {}))
            wadj = {"runs_mult": f.runs_mult, "hr_mult": f.hr_mult,
                    "wind_to_cf_mph": f.wind_to_cf_mph, "temp_f": f.temp_f}

            # Lineup-specific K% for pitcher K projection.
            # When the opposing lineup is confirmed, blend 65% lineup K% with
            # 35% team season K% — lineup K% better reflects today's actual
            # batters (e.g. rest days, bench players). Falls back to team K%
            # when lineup isn't posted yet.
            if opp_lineup_ids:
                lk = proj.lineup_k_pct(opp_lineup_ids, batter_stats)
                team_k = opp_off_idx.get("k_pct", 0.225)
                opp_off_idx = dict(opp_off_idx)
                opp_off_idx["k_pct"] = 0.65 * lk + 0.35 * team_k

            batters_out = []
            order = 1
            for bs in proj.get_likely_batters(tid, batter_stats, lineup_ids=lineup_ids):
                pid = int(bs.get("player_id") or 0)
                pl = proj.resolve_platoon(pid, sp_id_opp, bat_sides, pit_throws,
                                          bat_vs_l, bat_vs_r)
                rs = bat_recent.get(pid)
                p = proj.project_batter(bs, order, team_pred, opp_sp_q, park, wadj,
                                        recent_stats=rs,
                                        bat_side=pl["bat_side"],
                                        opp_pit_throws=pl["opp_pit_throws"],
                                        bat_split=pl["bat_split"],
                                        is_switch=pl.get("is_switch", False),
                                        sc_stats=sc_bat_data.get(pid))
                batters_out.append(asdict(p))
                order += 1

            if side == "away":
                gp.away_batters = batters_out
            else:
                gp.home_batters = batters_out

            if sp_id_self:
                ps = pitcher_stats.get(sp_id_self, {"player_id": sp_id_self, "name": "?", "team_id": tid})
                ps_recent = pit_recent.get(int(sp_id_self))
                pp = proj.project_pitcher(ps, tid, opp_off_idx, opp_pred, park, wadj,
                                          recent_stats=ps_recent)
                if side == "away":
                    gp.away_starter = asdict(pp)
                else:
                    gp.home_starter = asdict(pp)

        # Sportsbook lookup
        bk = _find_book(book_data, f.home_team, f.away_team) if book_data else None
        if bk is None and manual:
            bk = _find_book(manual.get("games", []), f.home_team, f.away_team)
        if bk is not None:
            gp.book = bk
            gp.book_source = odds_source if bk in book_data else "manual"
            game_value_all = value.evaluate_game_lines(
                f.home_team, f.away_team, home_pred, away_pred, bk,
                edge_threshold=-1.0,
            )
            for vb in game_value_all:
                vb.game_pk = int(f.game_pk)
            game_value = [vb for vb in game_value_all if vb.edge_pct >= edge_threshold * 100]
            gp.game_value = [_vb_to_dict(vb) for vb in game_value]
            gp.all_bets.extend(_vb_to_dict(vb) for vb in game_value_all)
            all_value_bets.extend(game_value)

        # Player props for this game
        if merged_props:
            by_name = {}
            for bs in (proj.get_likely_batters(f.home_team_id, batter_stats, lineup_ids=home_lineup_ids) +
                       proj.get_likely_batters(f.away_team_id, batter_stats, lineup_ids=away_lineup_ids)):
                by_name[bs.get("name", "")] = bs
            for sp_id in (f.home_sp_id, f.away_sp_id):
                if sp_id and pitcher_stats.get(sp_id):
                    by_name[pitcher_stats[sp_id].get("name", "")] = pitcher_stats[sp_id]

            our_props = [pp for pp in merged_props
                         if "game" not in pp
                         or (_team_name_match(pp["game"].split(" @ ")[0], f.away_team)
                             and _team_name_match(pp["game"].split(" @ ")[1], f.home_team))]

            game_prop_value = []
            for pp in our_props:
                name = pp.get("player", "")
                resolved = name_match.find_match(name, by_name.keys())
                pdata = by_name.get(resolved) if resolved else None
                if not pdata:
                    continue
                is_pitcher = pp["market"].startswith("pitcher_") or pp["market"] in ("outs", "ip")
                if is_pitcher:
                    is_home_pitcher = (pdata.get("player_id") == f.home_sp_id)
                    team_id = f.home_team_id if is_home_pitcher else f.away_team_id
                    opp_id = f.away_team_id if is_home_pitcher else f.home_team_id
                    opp_off = feats.team_offense_index(team_off.get(opp_id, {}))
                    # Use the confirmed opposing lineup's K% directly when
                    # available. lineup_k_pct already EB-shrinks each batter to
                    # league at PRIOR_PA=30, so a second shrinkage to team K%
                    # was double-dipping. Team K% additionally biases toward
                    # the team's bench/IL pool, which is the wrong prior when
                    # we know exactly who's hitting.
                    _opp_lineup = away_lineup_ids if is_home_pitcher else home_lineup_ids
                    if _opp_lineup:
                        opp_off = dict(opp_off)
                        opp_off["k_pct"] = proj.lineup_k_pct(_opp_lineup, batter_stats)
                    opp_pred = away_pred if is_home_pitcher else home_pred
                    pproj = proj.project_pitcher(pdata, team_id, opp_off, opp_pred, park,
                                                 {"runs_mult": f.runs_mult, "hr_mult": f.hr_mult},
                                                 recent_stats=pit_recent.get(int(pdata.get("player_id") or 0)))
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
                    pid = int(pdata.get("player_id") or 0)
                    pl = proj.resolve_platoon(pid, sp_id, bat_sides, pit_throws, bat_vs_l, bat_vs_r)
                    rs = bat_recent.get(pid)
                    bproj = proj.project_batter(pdata, 3, team_pred_local, opp_sp_q, park,
                                                {"runs_mult": f.runs_mult, "hr_mult": f.hr_mult,
                                                 "wind_to_cf_mph": f.wind_to_cf_mph, "temp_f": f.temp_f},
                                                recent_stats=rs,
                                                bat_side=pl["bat_side"],
                                                opp_pit_throws=pl["opp_pit_throws"],
                                                bat_split=pl["bat_split"],
                                                is_switch=pl.get("is_switch", False),
                                                sc_stats=sc_bat_data.get(pid))
                    means = {
                        "hr": bproj.proj_hr, "hits": bproj.proj_h, "tb": bproj.proj_tb,
                        "rbi": bproj.proj_rbi, "runs": bproj.proj_runs, "k": bproj.proj_k,
                        "bb": bproj.proj_bb, "sb": bproj.proj_sb,
                    }
                    mean = means.get(pp["market"])
                if mean is None:
                    continue
                vbs_all = value.evaluate_prop(name, pp["market"], mean, pp["line"],
                                              pp.get("over"), pp.get("under"),
                                              edge_threshold=-1.0)
                # Short-start guard: drop pitcher counting-stat OVERs when we
                # project < 4.5 IP (13.5 outs). Quick-hook starts — rookies on
                # pitch counts, openers, struggling vets — can collapse an
                # OVER pick to zero. The projection's NegBin distribution
                # doesn't capture this discrete-event risk well. May 2 example:
                # Lowder was a top-confidence OVER 4.5 K pick that went 1 K on
                # a quick pull. UNDERs are unaffected (a short start helps them).
                if is_pitcher and vbs_all and pproj.expected_outs < 14.5:
                    vbs_all = [vb for vb in vbs_all if " OVER " not in vb.description]
                if vbs_all:
                    _pid = int(pdata.get("player_id") or 0)
                    for vb in vbs_all:
                        vb.game_pk = int(f.game_pk)
                        vb.player_id = _pid
                    gp.all_bets.extend(_vb_to_dict(vb) for vb in vbs_all)
                    vbs = [vb for vb in vbs_all if vb.edge_pct >= edge_threshold * 100]
                    game_prop_value.extend(vbs)
                    all_value_bets.extend(vbs)
            gp.prop_value = [_vb_to_dict(vb) for vb in game_prop_value]

        games_out.append(gp)

    # Build slate-wide leaderboard. We rank by `score` (variance-adjusted edge)
    # rather than raw edge — this puts reliable, near-50/50 plays ahead of
    # high-variance lottery tickets with the same nominal edge.
    ranked = sorted(all_value_bets, key=lambda x: -getattr(x, "score", x.edge_pct))[:top_n]
    top_value = [_vb_to_dict(vb) for vb in ranked]

    # Concentration check: if a single game's bets occupy > 40% of the top
    # leaderboard, warn that it's effectively one bet.
    concentration_warning = None
    if ranked:
        # Tag each bet with the game it came from (best-effort match by team
        # names appearing in the description)
        team_names = []
        for gp in games_out:
            team_names.append((gp.home_team, gp.away_team, gp.game_pk))
        from collections import Counter
        bet_games = []
        for vb in ranked:
            desc_l = vb.description.lower()
            for h, a, pk in team_names:
                if h.lower() in desc_l or a.lower() in desc_l:
                    bet_games.append(pk)
                    break
        if bet_games:
            cnt = Counter(bet_games)
            top_pk, top_n_bets = cnt.most_common(1)[0]
            share = top_n_bets / len(ranked)
            if share >= 0.40:
                # Look up team names for the offending game
                gp = next((x for x in games_out if x.game_pk == top_pk), None)
                if gp:
                    concentration_warning = (
                        f"{top_n_bets} of the top {len(ranked)} value bets "
                        f"({share:.0%}) come from {gp.away_team} @ {gp.home_team}. "
                        f"They share one underlying signal — treat as ~1 position, not {top_n_bets}."
                    )

    # Log top confidence picks for outcome tracking (fire-and-forget)
    try:
        if top_value and target == datetime.now(timezone.utc).date():
            bet_tracker.log_picks(target, top_value, top_n=10)
            bet_tracker.evaluate_outcomes()
    except Exception:
        pass

    slate_all_bets: list[dict] = []
    for gp in games_out:
        slate_all_bets.extend(gp.all_bets)

    return SlateResult(
        target_date=target.isoformat(),
        odds_source=odds_source,
        n_games=len(games_out),
        n_books=len(book_data),
        n_props_loaded=len(merged_props),
        games=games_out,
        top_value=top_value,
        concentration_warning=concentration_warning,
        all_bets=slate_all_bets,
    )
