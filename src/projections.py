"""Player stat projections (batters and pitchers).

Given a `GameFeatures` row and a season-stat snapshot, project each likely
player's stats for the upcoming game.

Projections use rate stats × expected playing time × matchup adjustments.
Rates are shrunk toward league mean (empirical Bayes) so noisy April lines
don't dominate.

Calibration vector: this module exposes simple multiplicative adjustments
that the backtest tunes. We start with literature-informed values and let
the backtest validate.

ML-stacked refinement:
  When `data/models/prop_*.joblib` exist, we blend the analytical projection
  with a HistGBT trained on box-score residuals. Default blend = 50% analytical
  + 50% ML, which is empirically the best on the 2026 holdout. Set
  `ml_blend=0.0` to use pure analytical (the original behavior).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import math
import pandas as pd

# League rate priors (2024-2025 MLB averages)
LG = {
    "k_pct": 0.225,
    "bb_pct": 0.085,
    "hr_per_pa": 0.030,
    "hbp_per_pa": 0.012,
    "h_per_ab": 0.245,            # batting average
    "hr_per_ab": 0.034,
    "tb_per_ab": 0.395,
    "double_per_ab": 0.045,
    "triple_per_ab": 0.005,
    "rbi_per_pa": 0.115,
    "sb_per_g": 0.05,
    # Pitcher
    "k9": 8.7, "bb9": 3.2, "hr9": 1.20, "h9": 8.4,
}

# Empirical-Bayes shrinkage prior weights (in PA for batters, BF for pitchers).
# Tuned empirically on the 2026 player-prop calibration: prior=60 was pulling
# elite hitters' rates too far toward league mean, causing top-bin underestimation
# of ~20% across hits/HR/TB. With ~120 PA per starter at end of April,
# prior=30 gives elite hitters 80% weight on themselves, league 20%.
PRIOR_PA = 30.0
PRIOR_BF = 50.0


def _safe(x, default=0.0):
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _shrink(raw: float, league: float, n: float, prior_n: float) -> float:
    if n + prior_n <= 0:
        return league
    w = n / (n + prior_n)
    return w * raw + (1 - w) * league


# ---------- Lineup determination ----------
def get_likely_batters(team_id: int, batter_stats: dict[int, dict],
                       lineup_ids: list[int] | None = None,
                       max_n: int = 9) -> list[dict]:
    """Return likely starting batters for a team. If lineup is known, use it.
    Otherwise fall back to the team's PA leaderboard (top 9 by season PA).
    """
    if lineup_ids:
        out = []
        for pid in lineup_ids:
            s = batter_stats.get(pid) or batter_stats.get(str(pid))
            if s:
                out.append(s)
        if out:
            return out[:max_n]

    # Fall back to season PA leaderboard for that team
    pool = [s for s in batter_stats.values()
            if s.get("team_id") == team_id and _safe(s.get("plateAppearances")) >= 5]
    pool.sort(key=lambda s: -_safe(s.get("plateAppearances")))
    return pool[:max_n]


# ---------- Per-batter projection ----------
@dataclass
class BatterProjection:
    player_id: int
    name: str
    team_id: int
    bat_order: int                # 1-9, for PA expectation
    expected_pa: float
    proj_h: float
    proj_hr: float
    proj_2b: float
    proj_3b: float
    proj_tb: float
    proj_rbi: float
    proj_runs: float
    proj_k: float
    proj_bb: float
    proj_sb: float


def _expected_pa_by_order(order: int, runs_pred: float) -> float:
    """Estimate plate appearances based on lineup spot and run environment.

    A high-scoring game produces more PAs (around the order). Baseline ~4.3 PA
    for top of order, declining to ~3.7 for bottom. Each run above 4.5 lifts
    every spot ~0.07 PAs.
    """
    base = {1: 4.50, 2: 4.40, 3: 4.30, 4: 4.20, 5: 4.10,
            6: 4.00, 7: 3.90, 8: 3.80, 9: 3.70}
    pa = base.get(order, 4.0)
    pa += 0.07 * (runs_pred - 4.5)
    return max(2.5, min(5.5, pa))


# Lazy-loaded ML adjustment models. None means no models on disk; empty dict
# means no models for this kind. We load once and cache.
_PROP_MODELS: dict | None = None


def _get_prop_models() -> dict:
    global _PROP_MODELS
    if _PROP_MODELS is not None:
        return _PROP_MODELS
    try:
        from . import prop_models
        path = Path(__file__).resolve().parent.parent / "data" / "models"
        _PROP_MODELS = prop_models.load_all(path)
    except Exception:
        _PROP_MODELS = {"batter": {}, "pitcher": {}}
    return _PROP_MODELS


def reload_prop_models() -> None:
    """Reset the module-level prop-model cache so the next call to
    _get_prop_models() reloads from disk. Call this alongside
    st.cache_data.clear() so Streamlit's Refresh button also picks up
    freshly retrained .joblib files."""
    global _PROP_MODELS
    _PROP_MODELS = None


def _apply_ml_adjustment_batter(
    proj: "BatterProjection",
    bat_stats: dict,
    rec_stats: dict | None,
    opp_sp_q: dict,
    park,
    weather_adj: dict,
    team_pred_runs: float,
    blend: float | None = None,
) -> "BatterProjection":
    """Blend ML predictions with analytical projection.

    If `blend` is None, each StatModel's tuned blend_weight (set during
    training on the holdout) is used per stat. Pass an explicit float to
    override globally.
    """
    models = _get_prop_models().get("batter", {})
    if not models:
        return proj
    from . import prop_models as pm
    feat = pm.batter_feature_row(proj, bat_stats, rec_stats, opp_sp_q, park,
                                 weather_adj, team_pred_runs)
    df = pd.DataFrame([feat])
    new = BatterProjection(**asdict(proj))
    for stat, attr in [("h", "proj_h"), ("hr", "proj_hr"), ("tb", "proj_tb"),
                       ("rbi", "proj_rbi"), ("runs", "proj_runs"),
                       ("k", "proj_k"), ("bb", "proj_bb")]:
        if stat in models:
            w = blend if blend is not None else getattr(models[stat], "blend_weight", 0.5)
            if w <= 0:
                continue
            ml_pred = float(models[stat].predict(df)[0])
            anal = getattr(new, attr)
            setattr(new, attr, w * ml_pred + (1 - w) * anal)
    return new


def _apply_ml_adjustment_pitcher(
    proj: "PitcherProjection",
    pit_stats: dict,
    rec_stats: dict | None,
    opp_off_idx: dict,
    park,
    weather_adj: dict,
    opp_pred_runs: float,
    blend: float | None = None,
) -> "PitcherProjection":
    models = _get_prop_models().get("pitcher", {})
    if not models:
        return proj
    from . import prop_models as pm
    feat = pm.pitcher_feature_row(proj, pit_stats, rec_stats, opp_off_idx, park,
                                  weather_adj, opp_pred_runs)
    df = pd.DataFrame([feat])
    new = PitcherProjection(**asdict(proj))
    for stat, attr in [("k", "proj_k"), ("bb", "proj_bb"), ("h", "proj_h"),
                       ("er", "proj_er"), ("hr", "proj_hr_allowed"),
                       ("outs", "expected_outs")]:
        if stat in models:
            w = blend if blend is not None else getattr(models[stat], "blend_weight", 0.5)
            if w <= 0:
                continue
            ml_pred = float(models[stat].predict(df)[0])
            anal = getattr(new, attr)
            blended = w * ml_pred + (1 - w) * anal
            setattr(new, attr, blended)
    new.expected_ip = new.expected_outs / 3.0
    return new


def resolve_platoon(
    bat_pid: int,
    opp_sp_pid: int | None,
    bat_sides: dict[int, str],
    pit_throws: dict[int, str],
    bat_vs_l: dict[int, dict],
    bat_vs_r: dict[int, dict],
) -> dict:
    """Return {bat_side, opp_pit_throws, bat_split} for the matchup.

    Switch hitters: their effective batting side is opposite the pitcher's
    throwing hand (face RHP -> bat L; face LHP -> bat R).
    """
    pid_int = int(bat_pid) if bat_pid else 0
    pit_pid_int = int(opp_sp_pid) if opp_sp_pid else 0
    p_throws = pit_throws.get(pit_pid_int) or pit_throws.get(str(pit_pid_int)) if pit_pid_int else None
    raw_side = bat_sides.get(pid_int) or bat_sides.get(str(pid_int))
    if raw_side == "S":
        bat_side = "L" if p_throws == "R" else "R"
    else:
        bat_side = raw_side
    # Pick the relevant vs-hand split: if pitcher throws L -> bat_vs_l, else bat_vs_r.
    if p_throws == "L":
        split = bat_vs_l.get(pid_int) or bat_vs_l.get(str(pid_int))
    elif p_throws == "R":
        split = bat_vs_r.get(pid_int) or bat_vs_r.get(str(pid_int))
    else:
        split = None
    return {"bat_side": bat_side, "opp_pit_throws": p_throws, "bat_split": split}


def _platoon_multipliers(bat_side: str | None, pit_throws: str | None,
                         bat_split: dict | None,
                         bat_overall: dict) -> dict:
    """Return multiplicative adjustments for K%, BB%, HR%, AVG based on the
    matchup's split data.

    If we have a meaningful split sample (>= 30 PA) for the relevant
    pitcher hand, we compute the ratio between the player's vs-hand rate
    and their overall rate, then apply that ratio (shrunk toward 1.0).

    bat_side is the side they bat from FOR THIS AT-BAT (a switch hitter
    facing a RHP bats lefty -> "L"). pit_throws is 'L' or 'R'.
    """
    out = {"k": 1.0, "bb": 1.0, "hr": 1.0, "avg": 1.0}
    if not bat_split or not pit_throws:
        return out

    pa_split = _safe(bat_split.get("plateAppearances"))
    if pa_split < 30:           # too small to be informative
        return out

    pa_overall = _safe(bat_overall.get("plateAppearances"), 1.0) or 1.0
    ab_split = _safe(bat_split.get("atBats"))
    ab_overall = _safe(bat_overall.get("atBats"), 1.0) or 1.0

    # Compute vs-hand rates
    rates_split = {
        "k":   _safe(bat_split.get("strikeOuts")) / pa_split,
        "bb":  _safe(bat_split.get("baseOnBalls")) / pa_split,
        "hr":  _safe(bat_split.get("homeRuns")) / pa_split,
        "avg": (_safe(bat_split.get("hits")) / ab_split) if ab_split else 0.245,
    }
    rates_overall = {
        "k":   _safe(bat_overall.get("strikeOuts")) / pa_overall,
        "bb":  _safe(bat_overall.get("baseOnBalls")) / pa_overall,
        "hr":  _safe(bat_overall.get("homeRuns")) / pa_overall,
        "avg": (_safe(bat_overall.get("hits")) / ab_overall) if ab_overall else 0.245,
    }

    # Shrink the ratio toward 1.0 by sample size. At PA=200 we use 0.6 weight on
    # the ratio; at PA=30 we use 0.13. This caps platoon sample noise.
    w = pa_split / (pa_split + 130.0)
    for k, vsplit in rates_split.items():
        voverall = rates_overall[k] or 0.001
        ratio = vsplit / voverall
        # Cap absolute ratio to [0.6, 1.7] — extreme single-season samples
        # rarely reflect true platoon talent.
        ratio = max(0.6, min(1.7, ratio))
        out[k] = w * ratio + (1 - w) * 1.0
    return out


def project_batter(
    bat_stats: dict,
    bat_order: int,
    team_pred_runs: float,
    opp_sp_q: dict,
    park,
    weather_adj: dict,
    recent_stats: dict | None = None,
    ml_blend: float | None = None,
    bat_side: str | None = None,
    opp_pit_throws: str | None = None,
    bat_split: dict | None = None,
) -> BatterProjection:
    """Empirical-Bayes shrunk rates × expected PA × matchup multipliers.

    If `recent_stats` (last-14-day stats) is provided, season rates are
    blended with recent form using a weight that scales with recent PA.
    """
    pid = int(bat_stats.get("player_id") or bat_stats.get("id") or 0)
    name = bat_stats.get("name") or bat_stats.get("fullName", "")
    tid = int(bat_stats.get("team_id") or 0)

    # Blend season + recent form if we have enough recent PAs.
    if recent_stats:
        rec_pa = _safe(recent_stats.get("plateAppearances"))
        # Recent-form weight scales 0..0.4 as PA grows from 0 to 60. Caps at 0.4
        # so season info still anchors the projection.
        rw = min(0.4, rec_pa / 150.0)
        def _b(sk, rk=None):
            sv = _safe(bat_stats.get(sk))
            rv = _safe(recent_stats.get(rk or sk))
            return (1 - rw) * sv + rw * rv
        pa = _b("plateAppearances")
        ab = _b("atBats")
        h  = _b("hits")
        hr = _b("homeRuns")
        d  = _b("doubles")
        t  = _b("triples")
        bb = _b("baseOnBalls")
        k  = _b("strikeOuts")
        rbi= _b("rbi")
        runs_b = _b("runs")
        sb = _b("stolenBases")
    else:
        pa = _safe(bat_stats.get("plateAppearances"))
        ab = _safe(bat_stats.get("atBats"))
        h  = _safe(bat_stats.get("hits"))
        hr = _safe(bat_stats.get("homeRuns"))
        d  = _safe(bat_stats.get("doubles"))
        t  = _safe(bat_stats.get("triples"))
        bb = _safe(bat_stats.get("baseOnBalls"))
        k  = _safe(bat_stats.get("strikeOuts"))
        rbi= _safe(bat_stats.get("rbi"))
        runs_b = _safe(bat_stats.get("runs"))
        sb = _safe(bat_stats.get("stolenBases"))

    raw_avg = h / ab if ab else LG["h_per_ab"]
    raw_hr_pa = hr / pa if pa else LG["hr_per_pa"]
    raw_2b_pa = d / pa if pa else (LG["double_per_ab"] * 0.93)
    raw_3b_pa = t / pa if pa else (LG["triple_per_ab"] * 0.93)
    raw_bb_pa = bb / pa if pa else LG["bb_pct"]
    raw_k_pa = k / pa if pa else LG["k_pct"]
    raw_rbi_pa = rbi / pa if pa else LG["rbi_per_pa"]

    # Shrink each rate toward league average by PA
    avg = _shrink(raw_avg, LG["h_per_ab"], pa, PRIOR_PA)
    hr_pa = _shrink(raw_hr_pa, LG["hr_per_pa"], pa, PRIOR_PA)
    d_pa  = _shrink(raw_2b_pa, LG["double_per_ab"] * 0.93, pa, PRIOR_PA)
    t_pa  = _shrink(raw_3b_pa, LG["triple_per_ab"] * 0.93, pa, PRIOR_PA)
    bb_pa = _shrink(raw_bb_pa, LG["bb_pct"], pa, PRIOR_PA)
    k_pa  = _shrink(raw_k_pa, LG["k_pct"], pa, PRIOR_PA)
    rbi_pa = _shrink(raw_rbi_pa, LG["rbi_per_pa"], pa, PRIOR_PA)
    sb_pg = _shrink(sb / max(_safe(bat_stats.get("gamesPlayed"), 1), 1), LG["sb_per_g"],
                    _safe(bat_stats.get("gamesPlayed"), 0), 30.0)

    # Matchup multipliers from opposing pitcher (relative to league)
    sp_k9 = opp_sp_q.get("k9", LG["k9"])
    sp_bb9 = opp_sp_q.get("bb9", LG["bb9"])
    sp_hr9 = opp_sp_q.get("hr9", LG["hr9"])
    sp_era = opp_sp_q.get("era", 4.50)
    k_mult  = sp_k9 / LG["k9"]
    bb_mult = sp_bb9 / LG["bb9"]
    hr_mult_pitch = sp_hr9 / LG["hr9"]
    avg_mult = (sp_era / 4.50) ** 0.5     # better pitcher (lower ERA) => lower hits

    # Platoon adjustment: for switch hitters, batting side flips relative to
    # opposing pitcher. We expose `bat_side` already resolved by the caller
    # (predict.py / backtest.py / train_props.py).
    plat = _platoon_multipliers(bat_side, opp_pit_throws, bat_split, bat_stats)
    k_mult  *= plat["k"]
    bb_mult *= plat["bb"]
    hr_mult_pitch *= plat["hr"]
    avg_mult *= plat["avg"]

    # Park / weather
    pf_runs = park.pf_runs
    pf_hr = park.pf_hr
    runs_w = weather_adj.get("runs_mult", 1.0)
    hr_w = weather_adj.get("hr_mult", 1.0)

    # Expected playing time
    ePA = _expected_pa_by_order(bat_order, team_pred_runs)
    eAB = ePA * (1 - bb_pa)        # AB ~ PA * (1 - BB%) ignoring HBP/SF (~2-3% rounding)

    proj_h  = eAB * avg * avg_mult
    proj_hr = ePA * hr_pa * hr_mult_pitch * pf_hr * hr_w
    proj_2b = ePA * d_pa * pf_runs * runs_w
    proj_3b = ePA * t_pa * pf_runs * runs_w
    proj_singles = max(0.0, proj_h - proj_hr - proj_2b - proj_3b)
    proj_tb = proj_singles + 2 * proj_2b + 3 * proj_3b + 4 * proj_hr
    proj_k  = ePA * k_pa * k_mult
    proj_bb = ePA * bb_pa * bb_mult
    # RBI scales with team run environment relative to average
    proj_rbi = ePA * rbi_pa * (team_pred_runs / 4.5) * pf_runs * runs_w
    # Runs scored ~ getting on base * scoring rate, simplest model = team_runs / 9 * (OBP relative)
    obp_est = (avg + bb_pa) / (1 + bb_pa) if (1 + bb_pa) else 0.32
    proj_runs = (team_pred_runs / 9.0) * (obp_est / 0.320) * (ePA / 4.0)
    proj_sb = sb_pg

    out = BatterProjection(
        player_id=pid, name=name, team_id=tid, bat_order=bat_order,
        expected_pa=ePA,
        proj_h=proj_h, proj_hr=proj_hr, proj_2b=proj_2b, proj_3b=proj_3b,
        proj_tb=proj_tb, proj_rbi=proj_rbi, proj_runs=proj_runs,
        proj_k=proj_k, proj_bb=proj_bb, proj_sb=proj_sb,
    )
    # ml_blend=None -> use per-stat tuned weights; ml_blend=0 -> pure analytical
    if ml_blend is None or ml_blend > 0:
        out = _apply_ml_adjustment_batter(out, bat_stats, recent_stats,
                                          opp_sp_q, park, weather_adj,
                                          team_pred_runs, blend=ml_blend)
    return out


# ---------- Per-pitcher projection ----------
@dataclass
class PitcherProjection:
    player_id: int
    name: str
    team_id: int
    expected_outs: float
    expected_ip: float
    proj_k: float
    proj_bb: float
    proj_h: float
    proj_er: float
    proj_hr_allowed: float


def _expected_outs(pq: dict, opp_off_rpg: float) -> float:
    """Estimate IP for a starter.

    2024-2025 MLB starters average ~5.1 IP (15.3 outs) — a steady decline from
    the 6 IP era. We anchor to 15 outs (5.0 IP) for an average starter and
    adjust for FIP and opponent strength. Caps at 7.0 IP.
    """
    bf = pq.get("bf", 0)
    if bf < 30:        # rookie / minimal data — assume below-average outing
        return 13.5
    fip = pq.get("fip", 4.10)
    base_outs = 15.5 - (fip - 4.0) * 2.2     # FIP 3.0 -> ~17.7 outs, FIP 5.0 -> ~13.3
    base_outs -= max(0.0, opp_off_rpg - 4.5) * 0.7
    # Early-season pitch-count caps: starters with low BF count have shorter leashes
    if bf < 80:
        base_outs -= 1.0
    return max(8.0, min(21.0, base_outs))


def project_pitcher(
    pit_stats: dict,
    team_id: int,
    opp_off_idx: dict,
    opp_pred_runs: float,
    park,
    weather_adj: dict,
    recent_stats: dict | None = None,
    ml_blend: float | None = None,
) -> PitcherProjection:
    pid = int(pit_stats.get("player_id") or pit_stats.get("id") or 0)
    name = pit_stats.get("name") or pit_stats.get("fullName", "")

    from .features import pitcher_quality_index
    pq = pitcher_quality_index(pit_stats)
    # If we have meaningful recent BF (last 14d), blend recent pitcher quality
    # with season — typically captures sudden velocity drops, command issues.
    if recent_stats:
        rec_bf = _safe(recent_stats.get("battersFaced"))
        if rec_bf >= 30:
            pq_recent = pitcher_quality_index(recent_stats)
            rw = min(0.4, rec_bf / 200.0)
            for k in ("k9", "bb9", "whip", "era", "fip", "xfip", "hr9"):
                if k in pq and k in pq_recent:
                    pq[k] = (1 - rw) * pq[k] + rw * pq_recent[k]

    e_outs = _expected_outs(pq, opp_off_idx.get("rpg", 4.5))
    e_ip = e_outs / 3.0
    # Estimate batters faced for the start
    bf_per_inning = 4.30 + (pq["whip"] - 1.28) * 0.5    # higher WHIP -> more batters
    e_bf = max(e_ip * bf_per_inning, e_ip * 3.0)

    # Strikeouts: K/9 rate adjusted by opponent K%
    opp_k_factor = opp_off_idx.get("k_pct", 0.225) / 0.225
    proj_k = (pq["k9"] / 9.0) * e_ip * opp_k_factor

    # Walks: BB/9 adjusted by opponent BB%
    opp_bb_factor = opp_off_idx.get("bb_pct", 0.085) / 0.085
    proj_bb = (pq["bb9"] / 9.0) * e_ip * opp_bb_factor

    # Hits allowed: WHIP - BB rate
    proj_h = max(0.0, pq["whip"] * e_ip - proj_bb)
    proj_h *= (opp_off_idx.get("ops", 0.720) / 0.720) ** 0.5   # better offense -> more hits

    # HR allowed: HR/9 × park × weather × opponent ISO
    iso_factor = (opp_off_idx.get("iso", 0.150) / 0.150) ** 0.5
    proj_hr = (pq["hr9"] / 9.0) * e_ip * park.pf_hr * weather_adj.get("hr_mult", 1.0) * iso_factor

    # Earned runs: opp_pred_runs × (e_outs / 27) × FIP-relative quality
    # Starters typically allow ~85% of their share of total runs (defense + bullpen variance).
    fip_rel = pq["fip"] / 4.10
    proj_er = opp_pred_runs * (e_outs / 27.0) * (0.55 + 0.45 * fip_rel)

    out = PitcherProjection(
        player_id=pid, name=name, team_id=team_id,
        expected_outs=e_outs, expected_ip=e_ip,
        proj_k=proj_k, proj_bb=proj_bb, proj_h=proj_h,
        proj_er=proj_er, proj_hr_allowed=proj_hr,
    )
    if ml_blend is None or ml_blend > 0:
        out = _apply_ml_adjustment_pitcher(out, pit_stats, recent_stats,
                                           opp_off_idx, park, weather_adj,
                                           opp_pred_runs, blend=ml_blend)
    return out
