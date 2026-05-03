"""Streamlit web app for the MLB predictor.

Run:
    streamlit run app.py

Pages:
  - Slate (default): date picker, top value bets, per-game cards, drill-down
"""
from __future__ import annotations
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import predict_core, projections as proj, bet_tracker
from src import dispersion as disp_mod

_DISP_PATH = ROOT / "data" / "models" / "dispersion.json"
_TEAM_RUN_DISP = 1.4   # typical MLB over-dispersion for team scoring (var/mean)


# ---------- Page config ----------
st.set_page_config(
    page_title="MLB Predictor",
    page_icon=":baseball:",
    layout="wide",
)


# ---------- CI helpers ----------
def _ci(mu: float, phi: float, lo: float = 0.10, hi: float = 0.90):
    """80% CI (lo_count, hi_count) for NegBin(mean=mu, var/mean=phi).

    Falls back to normal approximation if scipy is unavailable.
    """
    if mu <= 0:
        return 0, 0
    phi = max(phi, 1.01)
    try:
        from scipy.stats import nbinom
        r = mu / (phi - 1.0)
        p_succ = r / (r + mu)
        return int(nbinom.ppf(lo, r, p_succ)), int(nbinom.ppf(hi, r, p_succ))
    except Exception:
        import math
        std = (phi * mu) ** 0.5
        return max(0, math.floor(mu - 1.28 * std)), math.ceil(mu + 1.28 * std)


def _ci_str(lo: int, hi: int) -> str:
    return f"{lo}–{hi}"


def _p_at_least_one(mu: float, phi: float) -> float:
    """P(X >= 1) = 1 - P(X = 0) under NegBin(mean=mu, var/mean=phi)."""
    if mu <= 0:
        return 0.0
    phi = max(phi, 1.01)
    try:
        from scipy.stats import nbinom
        r = mu / (phi - 1.0)
        p_succ = r / (r + mu)
        return float(1.0 - nbinom.pmf(0, r, p_succ))
    except Exception:
        import math
        return float(1.0 - math.exp(-mu))


@st.cache_data(ttl=3600, show_spinner=False)
def _load_disp_fits():
    return disp_mod.load_fits(_DISP_PATH)


def _batter_ci_rows(batters: list[dict], fits: dict) -> pd.DataFrame:
    """Build a DataFrame of batter projections with 80% CIs for each stat."""
    STAT_MAP = [
        ("H",  "proj_h",    "hits"),
        ("HR", "proj_hr",   "hr"),
        ("TB", "proj_tb",   "tb"),
        ("RBI","proj_rbi",  "rbi"),
        ("R",  "proj_runs", "runs"),
        ("K",  "proj_k",    "k"),
        ("BB", "proj_bb",   "bb"),
    ]
    rows = []
    for b in batters:
        row: dict = {"Player": b.get("name", "?"), "PA": round(b.get("expected_pa", 0), 1)}
        for label, proj_key, disp_key in STAT_MAP:
            mu = b.get(proj_key, 0) or 0
            phi = disp_mod.disp_for(disp_key, mu, fits)
            lo, hi = _ci(mu, phi)
            row[label]          = round(mu, 2)
            row[f"{label} CI"]  = _ci_str(lo, hi)
        rows.append(row)
    return pd.DataFrame(rows)


def _starter_ci_rows(starter: dict | None, fits: dict) -> pd.DataFrame:
    """Build a 1-row DataFrame for a starter with 80% CIs."""
    if not starter:
        return pd.DataFrame()
    STAT_MAP = [
        ("K",   "proj_k",          "pitcher_k"),
        ("IP",  "expected_ip",     "pitcher_outs"),   # dispersion fit on outs; IP = outs/3
        ("ER",  "proj_er",         "pitcher_er"),
        ("H",   "proj_h",          "pitcher_h"),
        ("BB",  "proj_bb",         "pitcher_bb"),
        ("HR",  "proj_hr_allowed", "pitcher_hr"),
    ]
    row: dict = {"Pitcher": starter.get("name", "?")}
    for label, proj_key, disp_key in STAT_MAP:
        mu = starter.get(proj_key, 0) or 0
        if disp_key == "pitcher_outs":
            # dispersion fit on expected_outs; convert to IP for display
            mu_outs = mu * 3
            phi = disp_mod.disp_for(disp_key, mu_outs, fits)
            lo_o, hi_o = _ci(mu_outs, phi)
            row[label]         = round(mu, 2)
            row[f"{label} CI"] = _ci_str(round(lo_o / 3, 1), round(hi_o / 3, 1))
        else:
            phi = disp_mod.disp_for(disp_key, mu, fits)
            lo, hi = _ci(mu, phi)
            row[label]         = round(mu, 2)
            row[f"{label} CI"] = _ci_str(lo, hi)
    return pd.DataFrame([row])


# ---------- Formatting helpers ----------
def _amer(v):
    if v is None:
        return "?"
    return f"+{v}" if v > 0 else str(v)


def _render_value_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.copy()
    df["odds"] = df["odds"].apply(_amer)
    df["model"] = (df["model_prob"] * 100).round(1).astype(str) + "%"
    df["no-vig"] = (df["novig_prob"] * 100).round(1).astype(str) + "%"
    df["edge"] = "+" + df["edge_pct"].round(1).astype(str) + "%"
    df["EV/$"] = df["ev_per_dollar"].apply(lambda x: f"{x:+.3f}")
    df["Kelly"] = (df["kelly"] * 100).round(2).astype(str) + "%"
    if "score" in df.columns:
        df["Score"] = df["score"].round(2)
    cols = ["description", "odds", "model", "no-vig", "edge"]
    if "Score" in df.columns:
        cols.append("Score")
    cols += ["EV/$", "Kelly"]
    return df[cols].rename(columns={"description": "Bet"})


def _render_batter_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.copy()
    keep = ["name", "expected_pa", "proj_h", "proj_hr", "proj_tb",
            "proj_rbi", "proj_runs", "proj_k", "proj_bb"]
    df = df[keep].rename(columns={
        "name": "Player", "expected_pa": "PA",
        "proj_h": "H", "proj_hr": "HR", "proj_tb": "TB",
        "proj_rbi": "RBI", "proj_runs": "R",
        "proj_k": "K", "proj_bb": "BB",
    })
    for c in ["PA", "H", "HR", "TB", "RBI", "R", "K", "BB"]:
        df[c] = df[c].round(2)
    return df


def _render_pitcher_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.copy()
    keep = ["name", "expected_ip", "proj_k", "proj_bb", "proj_h", "proj_er", "proj_hr_allowed"]
    df = df[keep].rename(columns={
        "name": "Pitcher", "expected_ip": "IP",
        "proj_k": "K", "proj_bb": "BB", "proj_h": "H",
        "proj_er": "ER", "proj_hr_allowed": "HR",
    })
    for c in ["IP", "K", "BB", "H", "ER", "HR"]:
        df[c] = df[c].round(2)
    return df


# ---------- Cached prediction call ----------
@st.cache_data(ttl=600, show_spinner=False)
def run_prediction(target_iso: str, edge: float, fetch_odds: bool):
    res = predict_core.predict_slate(
        target_date=target_iso,
        edge_threshold=edge,
        fetch_odds=fetch_odds,
        top_n=60,
    )
    return res


# ---------- Sidebar controls ----------
with st.sidebar:
    st.title(":baseball: MLB Predictor")
    st.caption("Live model + Bovada lines • free, no API key required")

    today = datetime.now(timezone.utc).date()
    selected_date = st.date_input(
        "Game date",
        value=today,
        min_value=today - timedelta(days=14),
        max_value=today + timedelta(days=14),
    )
    edge_threshold = st.slider(
        "Min edge to flag",
        min_value=0.01, max_value=0.20, value=0.04, step=0.01,
        format="%.2f",
        help="Only show bets where model probability beats no-vig market by at least this much.",
    )
    fetch_odds = st.checkbox("Pull live odds (Bovada)", value=True)
    ranking_mode = st.radio(
        "Rank leaderboard by",
        ["Score", "EV / $"],
        horizontal=True,
        help=(
            "**Score** = (edge / √(p·(1−p))) × stat-reliability — Sharpe-like; favours high-confidence edges. "
            "**EV / $** = expected profit per dollar wagered (raw model value)."
        ),
    )
    if st.button(":arrows_counterclockwise: Refresh"):
        run_prediction.clear()
        proj.reload_prop_models()

    st.divider()
    st.caption(
        "Built on the MLB Stats API, Open-Meteo, and Bovada's open JSON feed. "
        "Backtested on 2026 season-to-date games."
    )


# ---------- Header ----------
st.title("MLB Slate")
st.caption(f"Predictions for **{selected_date.isoformat()}**")

with st.spinner("Pulling slate, weather, lines, projections..."):
    try:
        slate = run_prediction(selected_date.isoformat(), edge_threshold, fetch_odds)
    except FileNotFoundError as e:
        st.error(
            f"Missing data file: `{e.filename}`. Run the build pipeline first:\n\n"
            "```\npython -m scripts.build_dataset\npython -m scripts.train\npython -m scripts.train_props\npython -m scripts.fit_dispersion\n```"
        )
        st.stop()

if slate.n_games == 0:
    st.info("No games scheduled for this date.")
    st.stop()


# ---------- Top metrics ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Games", slate.n_games)
m2.metric("Books loaded", slate.n_books)
m3.metric("Player props", slate.n_props_loaded)
m4.metric("Top value bets", len(slate.top_value))

if slate.odds_source != "none":
    st.caption(f"Odds source: **{slate.odds_source}**")

if slate.concentration_warning:
    st.warning(":warning: " + slate.concentration_warning)


# ---------- Main tabs ----------
sort_key = "score" if ranking_mode == "Score" else "ev_per_dollar"
_sort_col = "Score" if sort_key == "score" else "EV/$"

_all_bets: list[dict] = []
for _gp in slate.games:
    _all_bets.extend(_gp.game_value)
    _all_bets.extend(_gp.prop_value)

main_tab_value, main_tab_intervals, main_tab_p1, main_tab_games, main_tab_track = st.tabs(
    ["Value Bets", "Intervals", "P(≥1) Confidence", "Games", "Track Record"]
)


# ===== TAB 1 — Value Bets leaderboard =====
with main_tab_value:
    if ranking_mode == "Score":
        st.caption(
            "Ranked by **Score** = (edge / √(p·(1−p))) × stat-reliability — Sharpe-like; "
            "favours high-confidence edges over longshot lottery tickets."
        )
    else:
        st.caption(
            "Ranked by **EV / $** = expected profit per dollar wagered — raw model value "
            "with no penalty for longshots or low-reliability markets."
        )

    MARKET_TABS = [
        ("Overall",     None),
        ("Confidence",  "__confidence__"),
        ("HR",          ["prop_hr"]),
        ("Hits",        ["prop_hits"]),
        ("Total Bases", ["prop_tb"]),
        ("RBI",         ["prop_rbi"]),
        ("Runs",        ["prop_runs"]),
        ("Batter K",    ["prop_k"]),
        ("Walks",       ["prop_bb"]),
        ("Pitcher K",   ["prop_pitcher_k"]),
        ("Pitcher",     ["prop_pitcher_outs", "prop_pitcher_er", "prop_pitcher_h", "prop_pitcher_bb", "prop_pitcher_hr"]),
        ("Game Lines",  ["moneyline", "total", "run_line"]),
    ]

    if not slate.top_value and not _all_bets:
        st.info("No value bets exceed the edge threshold. Lower the slider on the left to see more.")
    else:
        tabs = st.tabs([t[0] for t in MARKET_TABS])
        for tab, (label, markets) in zip(tabs, MARKET_TABS):
            with tab:
                if markets is None:
                    pool = sorted(slate.top_value, key=lambda x: -x.get(sort_key, 0))[:40]
                elif markets == "__confidence__":
                    pool = sorted(_all_bets, key=lambda x: -(x.get("confidence") or 0))[:20]
                else:
                    pool = sorted(
                        [b for b in _all_bets if b.get("market") in markets],
                        key=lambda x: -x.get(sort_key, 0),
                    )
                if not pool:
                    st.info(f"No value bets for {label}.")
                    continue

                is_confidence_tab = (markets == "__confidence__")
                if is_confidence_tab:
                    st.caption(
                        "**Confidence** = stat-reliability × 4·p·(1−p) — peaks at even-money bets "
                        "backed by high-reliability markets. Top picks here are logged daily for outcome tracking."
                    )

                _raw = pd.DataFrame(pool)
                _df_cols: dict = {
                    "Bet":        _raw["description"],
                    "Odds":       _raw["odds"].apply(_amer),
                    "Model%":     (_raw["model_prob"] * 100).round(1),
                    "No-vig%":    (_raw["novig_prob"] * 100).round(1),
                    "Edge%":      _raw["edge_pct"].round(1),
                    "Confidence": _raw["confidence"].round(3) if "confidence" in _raw.columns else 0.0,
                    "Score":      _raw["score"].round(3),
                    "EV/$":       _raw["ev_per_dollar"].round(4),
                    "Kelly%":     (_raw["kelly"] * 100).round(3),
                }
                _sort_for_tab = "Confidence" if is_confidence_tab else _sort_col
                df_lb = pd.DataFrame(_df_cols).sort_values(_sort_for_tab, ascending=False)

                st.dataframe(
                    df_lb,
                    column_config={
                        "Model%":     st.column_config.NumberColumn("Model%",     format="%.1f%%"),
                        "No-vig%":    st.column_config.NumberColumn("No-vig%",    format="%.1f%%"),
                        "Edge%":      st.column_config.NumberColumn("Edge%",      format="+%.1f%%"),
                        "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                        "Score":      st.column_config.NumberColumn("Score",      format="%.2f"),
                        "EV/$":       st.column_config.NumberColumn("EV/$",       format="+%.3f"),
                        "Kelly%":     st.column_config.NumberColumn("Kelly%",     format="%.2f%%"),
                    },
                    use_container_width=True, hide_index=True,
                    height=min(500, 38 * len(df_lb) + 38),
                )


# ===== TAB 2 — Confidence Intervals =====
with main_tab_intervals:
    st.caption(
        "**80% confidence intervals** (10th–90th percentile) for every player stat and game total. "
        "Computed from the model's projected mean using empirically-fitted NegBin dispersion per stat. "
        "A narrow CI means the outcome is fairly predictable; a wide CI means high variance."
    )

    fits = _load_disp_fits()

    # --- Section 1: Game totals ---
    st.subheader("Game totals")

    game_rows = []
    for gp in slate.games:
        lo_a, hi_a = _ci(gp.pred_away_runs, _TEAM_RUN_DISP)
        lo_h, hi_h = _ci(gp.pred_home_runs, _TEAM_RUN_DISP)
        lo_t, hi_t = _ci(gp.pred_total, _TEAM_RUN_DISP * 0.9)  # totals slightly less dispersed
        game_rows.append({
            "Matchup":      f"{gp.away_team} @ {gp.home_team}",
            "Away pred":    round(gp.pred_away_runs, 2),
            "Away 80%CI":   _ci_str(lo_a, hi_a),
            "Home pred":    round(gp.pred_home_runs, 2),
            "Home 80%CI":   _ci_str(lo_h, hi_h),
            "Total pred":   round(gp.pred_total, 2),
            "Total 80%CI":  _ci_str(lo_t, hi_t),
            "P(home win)":  round(gp.p_home_win * 100, 1),
            "P(over 8.5)":  round(gp.p_over_8_5 * 100, 1),
        })

    game_df = pd.DataFrame(game_rows)
    st.dataframe(
        game_df,
        column_config={
            "Away pred":   st.column_config.NumberColumn("Away pred",  format="%.2f"),
            "Home pred":   st.column_config.NumberColumn("Home pred",  format="%.2f"),
            "Total pred":  st.column_config.NumberColumn("Total pred", format="%.2f"),
            "P(home win)": st.column_config.NumberColumn("P(home win)", format="%.1f%%"),
            "P(over 8.5)": st.column_config.NumberColumn("P(over 8.5)", format="%.1f%%"),
        },
        use_container_width=True, hide_index=True,
    )

    # --- Section 2: Player intervals — all games ---
    st.subheader("Player stat intervals")

    def _show_batter_intervals(batters: list[dict], team: str):
        if not batters:
            st.caption(f"No batter data for {team}.")
            return
        df = _batter_ci_rows(batters, fits)
        if df.empty:
            return
        num_cfg = {c: st.column_config.NumberColumn(c, format="%.2f")
                   for c in ["PA", "H", "HR", "TB", "RBI", "R", "K", "BB"]}
        st.dataframe(df, column_config=num_cfg,
                     use_container_width=True, hide_index=True,
                     height=min(600, 38 * len(df) + 38))

    def _show_starter_intervals(starter: dict | None, team: str):
        if not starter:
            st.caption(f"No starter data for {team}.")
            return
        df = _starter_ci_rows(starter, fits)
        if df.empty:
            return
        num_cfg = {c: st.column_config.NumberColumn(c, format="%.2f")
                   for c in ["K", "IP", "ER", "H", "BB", "HR"]}
        st.dataframe(df, column_config=num_cfg,
                     use_container_width=True, hide_index=True)

    for gp in slate.games:
        matchup_label = f"{gp.away_team} @ {gp.home_team}  —  {gp.pred_away_runs:.1f}–{gp.pred_home_runs:.1f} pred"
        with st.expander(matchup_label, expanded=True):
            bat_c1, bat_c2 = st.columns(2)
            with bat_c1:
                st.markdown(f"**{gp.away_team} batters**")
                _show_batter_intervals(gp.away_batters, gp.away_team)
                st.markdown(f"**{gp.away_team} starter:** {gp.away_sp_name or '?'}")
                _show_starter_intervals(gp.away_starter, gp.away_team)
            with bat_c2:
                st.markdown(f"**{gp.home_team} batters**")
                _show_batter_intervals(gp.home_batters, gp.home_team)
                st.markdown(f"**{gp.home_team} starter:** {gp.home_sp_name or '?'}")
                _show_starter_intervals(gp.home_starter, gp.home_team)

    # --- Section 3: CI width summary (which bets are most certain?) ---
    st.subheader("Narrowest intervals — highest certainty bets")
    st.caption(
        "Bets where the model's 80% CI is tightest relative to the prop line. "
        "A narrow spread (hi - lo) vs a book line at, say, 0.5 means the outcome is predictable."
    )

    width_rows = []
    for gp in slate.games:
        for b in gp.away_batters + gp.home_batters:
            for label, proj_key, disp_key in [
                ("H",  "proj_h",    "hits"),
                ("HR", "proj_hr",   "hr"),
                ("K",  "proj_k",    "k"),
                ("TB", "proj_tb",   "tb"),
            ]:
                mu = b.get(proj_key, 0) or 0
                if mu <= 0:
                    continue
                phi = disp_mod.disp_for(disp_key, mu, fits)
                lo, hi = _ci(mu, phi)
                # Skip if the entire 80% CI is zero — stat is effectively never occurring
                if hi == 0:
                    continue
                width_rows.append({
                    "Player":   b.get("name", "?"),
                    "Team":     gp.away_team if b in gp.away_batters else gp.home_team,
                    "Stat":     label,
                    "Proj":     round(mu, 2),
                    "CI low":   lo,
                    "CI high":  hi,
                    "CI width": hi - lo,
                })

    if width_rows:
        width_df = (
            pd.DataFrame(width_rows)
            .sort_values("CI width")
            .reset_index(drop=True)
        )
        st.dataframe(
            width_df,
            column_config={
                "Proj":     st.column_config.NumberColumn("Proj",     format="%.2f"),
                "CI low":   st.column_config.NumberColumn("CI low",   format="%d"),
                "CI high":  st.column_config.NumberColumn("CI high",  format="%d"),
                "CI width": st.column_config.NumberColumn("CI width", format="%d"),
            },
            use_container_width=True, hide_index=True,
        )


# ===== TAB 3 — P(≥1) Confidence =====
with main_tab_p1:
    st.caption(
        "Props ranked by how confident the model is that the player achieves **at least 1** of that stat. "
        "Computed from the projected mean using empirically-fitted NegBin dispersion per stat. "
        "Useful for finding 'safe' Yes props (HR, Hits, K, etc.) where the model sees a high floor."
    )

    fits_p1 = _load_disp_fits()

    BATTER_STATS_P1 = [
        ("H",   "proj_h",    "hits"),
        ("HR",  "proj_hr",   "hr"),
        ("TB",  "proj_tb",   "tb"),
        ("RBI", "proj_rbi",  "rbi"),
        ("R",   "proj_runs", "runs"),
        ("K",   "proj_k",    "k"),
        ("BB",  "proj_bb",   "bb"),
    ]
    PITCHER_STATS_P1 = [
        ("K",  "proj_k",   "pitcher_k"),
        ("H",  "proj_h",   "pitcher_h"),
        ("BB", "proj_bb",  "pitcher_bb"),
        ("ER", "proj_er",  "pitcher_er"),
    ]

    p1_rows = []
    for gp in slate.games:
        matchup = f"{gp.away_team} @ {gp.home_team}"

        for b in gp.away_batters + gp.home_batters:
            team = gp.away_team if b in gp.away_batters else gp.home_team
            for label, proj_key, disp_key in BATTER_STATS_P1:
                mu = b.get(proj_key, 0) or 0
                if mu <= 0:
                    continue
                phi = disp_mod.disp_for(disp_key, mu, fits_p1)
                p1 = _p_at_least_one(mu, phi)
                _, hi = _ci(mu, phi)
                if hi == 0:
                    continue  # 90th pct is still 0 — skip
                p1_rows.append({
                    "Player":   b.get("name", "?"),
                    "Team":     team,
                    "Matchup":  matchup,
                    "Stat":     label,
                    "Proj":     round(mu, 2),
                    "P(≥1)":    round(p1 * 100, 1),
                })

        for starter, team in [(gp.away_starter, gp.away_team), (gp.home_starter, gp.home_team)]:
            if not starter:
                continue
            for label, proj_key, disp_key in PITCHER_STATS_P1:
                mu = starter.get(proj_key, 0) or 0
                if mu <= 0:
                    continue
                phi = disp_mod.disp_for(disp_key, mu, fits_p1)
                p1 = _p_at_least_one(mu, phi)
                _, hi = _ci(mu, phi)
                if hi == 0:
                    continue
                p1_rows.append({
                    "Player":   starter.get("name", "?") + " (SP)",
                    "Team":     team,
                    "Matchup":  matchup,
                    "Stat":     label,
                    "Proj":     round(mu, 2),
                    "P(≥1)":    round(p1 * 100, 1),
                })

    if not p1_rows:
        st.info("No projection data available.")
    else:
        p1_df = pd.DataFrame(p1_rows).sort_values("P(≥1)", ascending=False).reset_index(drop=True)

        # Stat filter
        all_stats = sorted(p1_df["Stat"].unique())
        sel_stats = st.multiselect(
            "Filter by stat", all_stats, default=all_stats, key="p1_stat_filter"
        )
        if sel_stats:
            p1_df = p1_df[p1_df["Stat"].isin(sel_stats)]

        # Confidence threshold slider
        min_p1 = st.slider(
            "Min P(≥1)%", min_value=50, max_value=99, value=70, step=5,
            key="p1_thresh",
            help="Only show props where the model gives at least this probability of ≥1 occurring.",
        )
        p1_df = p1_df[p1_df["P(≥1)"] >= min_p1].reset_index(drop=True)

        if p1_df.empty:
            st.info("No props meet the threshold. Lower the slider.")
        else:
            st.dataframe(
                p1_df,
                column_config={
                    "Proj":  st.column_config.NumberColumn("Proj",  format="%.2f"),
                    "P(≥1)": st.column_config.NumberColumn("P(≥1)", format="%.1f%%"),
                },
                use_container_width=True,
                hide_index=True,
                height=min(700, 38 * len(p1_df) + 38),
            )


# ===== TAB 4 — Per-game cards =====
with main_tab_games:
    for gp in slate.games:
        fav = gp.home_team if gp.p_home_win >= 0.5 else gp.away_team
        fav_pct = gp.p_home_win if fav == gp.home_team else (1 - gp.p_home_win)
        summary = (
            f"**{gp.away_team}** @ **{gp.home_team}**  •  "
            f"{gp.pred_away_runs:.1f} - {gp.pred_home_runs:.1f}  •  "
            f"{fav} {fav_pct:.0%}"
        )
        n_value = len(gp.game_value) + len(gp.prop_value)
        if n_value:
            summary += f"  •  :money_with_wings: {n_value} value bet(s)"

        with st.expander(summary, expanded=False):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                roof = f" ({gp.park_roof})" if gp.park_roof != "open" else ""
                local_pitch = ""
                try:
                    d = datetime.fromisoformat(gp.first_pitch_utc)
                    local_pitch = d.strftime("%H:%M UTC")
                except Exception:
                    pass
                st.markdown(f"**Venue:** {gp.venue}{roof}")
                st.markdown(f"**First pitch:** {local_pitch}")
                st.markdown(f"**Park factors:** runs {gp.park_pf_runs:.2f}, HR {gp.park_pf_hr:.2f}")
            with c2:
                st.markdown(f"**Weather (game time):**")
                st.markdown(f"&nbsp;&nbsp;{gp.temp_f:.0f}°F • wind to CF {gp.wind_to_cf_mph:+.1f} mph")
                st.markdown(f"&nbsp;&nbsp;runs ×{gp.runs_mult:.2f} • HR ×{gp.hr_mult:.2f}")
            with c3:
                st.markdown(f"**Starters (FIP / xFIP):**")
                st.markdown(f"&nbsp;&nbsp;{gp.away_sp_name or '?'} ({gp.away_sp_fip:.2f} / {gp.away_sp_xfip:.2f})")
                st.markdown(f"&nbsp;&nbsp;{gp.home_sp_name or '?'} ({gp.home_sp_fip:.2f} / {gp.home_sp_xfip:.2f})")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Predicted total", f"{gp.pred_total:.2f}")
            p2.metric(f"{gp.away_team[:18]}", f"{gp.pred_away_runs:.2f}")
            p3.metric(f"{gp.home_team[:18]}", f"{gp.pred_home_runs:.2f}")
            p4.metric("P(over 8.5)", f"{gp.p_over_8_5:.0%}")

            if gp.book:
                ml = (gp.book.get("moneyline") or {})
                tot = (gp.book.get("total") or {})
                rl = (gp.book.get("run_line") or {})
                st.markdown(
                    f"**Lines ({gp.book_source}):** "
                    f"ML {gp.away_team[:8]} {_amer(ml.get('away'))} / "
                    f"{gp.home_team[:8]} {_amer(ml.get('home'))}  •  "
                    f"Total {tot.get('line', '?')} (O {_amer(tot.get('over'))} / U {_amer(tot.get('under'))})  •  "
                    f"RL ±{abs(rl.get('line', 1.5)):.1f}"
                )

            if gp.game_value:
                st.markdown("**Game-line value:**")
                st.dataframe(_render_value_df(gp.game_value), use_container_width=True, hide_index=True)
            if gp.prop_value:
                st.markdown("**Player prop value:**")
                st.dataframe(_render_value_df(gp.prop_value), use_container_width=True, hide_index=True)

            st.markdown("---")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(f"**{gp.away_team} batters**")
                st.dataframe(_render_batter_df(gp.away_batters),
                             use_container_width=True, hide_index=True)
                if gp.away_starter:
                    st.markdown(f"**Starter:** {gp.away_starter['name']}")
                    st.dataframe(_render_pitcher_df([gp.away_starter]),
                                 use_container_width=True, hide_index=True)
            with b2:
                st.markdown(f"**{gp.home_team} batters**")
                st.dataframe(_render_batter_df(gp.home_batters),
                             use_container_width=True, hide_index=True)
                if gp.home_starter:
                    st.markdown(f"**Starter:** {gp.home_starter['name']}")
                    st.dataframe(_render_pitcher_df([gp.home_starter]),
                                 use_container_width=True, hide_index=True)


# ===== TAB 5 — Track Record =====
with main_tab_track:
    st.caption("Top-10 confidence picks logged automatically each day. Outcomes evaluated against final boxscores.")

    try:
        _n_updated = bet_tracker.evaluate_outcomes()
        _record = bet_tracker.get_track_record(days=30)

        if _record["total"] == 0:
            st.info("No picks logged yet. Run predictions for today's slate to start tracking.")
        else:
            _decided = _record["wins"] + _record["losses"]
            _wr = _record["win_rate"]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Picks logged", _record["total"])
            r2.metric("Decided", _decided)
            r3.metric("Wins", _record["wins"])
            r4.metric(
                "Win rate",
                f"{_wr:.1%}" if _wr is not None else "—",
                delta=f"{(_wr - 0.5)*100:+.1f}pp vs 50%" if _wr is not None else None,
            )

            if _record["by_market"]:
                _mkt_rows = []
                for mkt, bm in sorted(_record["by_market"].items()):
                    dec = bm["wins"] + bm["losses"]
                    _mkt_rows.append({
                        "Market":  mkt,
                        "Total":   bm["total"],
                        "W":       bm["wins"],
                        "L":       bm["losses"],
                        "Pending": bm["pending"],
                        "Win%":    f"{bm['wins']/dec:.0%}" if dec else "—",
                    })
                st.dataframe(pd.DataFrame(_mkt_rows), use_container_width=True, hide_index=True)

            with st.expander("Recent pick log", expanded=False):
                _log_rows = []
                for e in _record["entries"][:50]:
                    outcome = e.get("outcome") or "pending"
                    _log_rows.append({
                        "Date":       e["date"],
                        "Bet":        e["description"],
                        "Market":     e["market"],
                        "Line":       e.get("line", ""),
                        "Odds":       _amer(e.get("odds")) if e.get("odds") else "?",
                        "Model%":     f"{float(e.get('model_prob', 0))*100:.1f}%",
                        "Confidence": f"{float(e.get('confidence', 0)):.3f}",
                        "Outcome":    outcome,
                        "Actual":     e.get("actual", ""),
                    })
                if _log_rows:
                    _log_df = pd.DataFrame(_log_rows)
                    st.dataframe(_log_df, use_container_width=True, hide_index=True,
                                 height=min(600, 38 * len(_log_df) + 38))
    except Exception as _e:
        st.info(f"Track record unavailable: {_e}")
