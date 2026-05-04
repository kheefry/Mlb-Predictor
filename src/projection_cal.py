"""Per-stat isotonic calibration for player projections.

The 21-day backtest showed every top-decile counting stat under-projects
and every bottom-decile over-projects — a uniform compression toward the
mean across batter and pitcher markets. Fitting a monotonic curve
`actual = f(projected)` per stat and applying it at predict time bends the
curve back to observed reality, attacking the compression in one shot.

Calibrators live at data/models/projection_calibration.joblib (a dict of
sklearn IsotonicRegression objects keyed by stat name like 'pitcher_k',
'batter_tb', etc.). Fitting is done by scripts/fit_projection_calibration.py
on holdout backtest data; loading happens once per session and the result
is cached.

If the calibration file is missing or a stat key isn't present, we return
the raw projection unchanged — safe-fallback behaviour. This means we can
ship the apply() integration before any calibrators exist on disk.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

# Lazy-loaded cache
_CALIBRATORS: Optional[dict] = None
_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "projection_calibration.joblib"


def load() -> dict:
    """Load the per-stat isotonic calibrators from disk; return empty if absent."""
    global _CALIBRATORS
    if _CALIBRATORS is not None:
        return _CALIBRATORS
    try:
        import joblib
        if _PATH.exists():
            _CALIBRATORS = joblib.load(_PATH)
        else:
            _CALIBRATORS = {}
    except Exception:
        _CALIBRATORS = {}
    return _CALIBRATORS


def reload() -> None:
    """Reset the cache so the next call to load() reloads from disk."""
    global _CALIBRATORS
    _CALIBRATORS = None


def apply(stat: str, raw_proj: float) -> float:
    """Apply the calibrated mapping for `stat` if available, else passthrough.

    Calibrator keys are like 'batter_h', 'batter_hr', 'pitcher_k', etc.
    """
    if raw_proj is None:
        return raw_proj
    cals = load()
    iso = cals.get(stat)
    if iso is None:
        return float(raw_proj)
    try:
        out = float(iso.predict([raw_proj])[0])
    except Exception:
        return float(raw_proj)
    # Guard against negative or absurd outputs from a degenerate fit.
    return max(0.0, out)
