"""Fit empirical dispersion curves and save them to data/models/dispersion.json.

Run after train_props.py (which produces props_*_2026.csv).

Run: python -m scripts.fit_dispersion
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import dispersion as disp


BAT_CSV = ROOT / "data" / "games" / "props_bat_2026.csv"
PIT_CSV = ROOT / "data" / "games" / "props_pit_2026.csv"
OUT = ROOT / "data" / "models" / "dispersion.json"


def main():
    print("Fitting dispersion curves...")
    fits = disp.fit_all(BAT_CSV, PIT_CSV)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fits, indent=2), encoding="utf-8")

    print(f"Saved -> {OUT}\n")

    # Print summary
    for kind, group in fits.items():
        print(f"=== {kind.upper()} dispersions ===")
        for stat, d in group.items():
            edges = ["%.2f" % e if e != float("inf") else "inf" for e in d["bin_edges"]]
            disps = ["%.2f" % v for v in d["dispersions"]]
            print(f"  {stat:12s}  bins={edges}  disp={disps}  overall={d['overall']:.2f}")
        print()


if __name__ == "__main__":
    main()
