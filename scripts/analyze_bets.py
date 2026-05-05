"""Analyze bet log outcomes by market, and simulate the three new guards.

Guards simulated (applied to historical log to estimate impact):
  1. K OVER guard       — drop K OVERs with edge < 15%
  2. Runs OVER position — drop runs OVERs where batter is lineup spot 6-9
                          (approximated by description containing known bottom-order
                          names from the log — direct position data not in log)
  3. K UNDER low-line   — drop K UNDERs where line <= 4.5 AND
                          we can infer the model didn't project >= 1 K below line
                          (proxy: actual > line by >= 1, i.e. clear miss on a low line)

Guards 1 and 3 can be applied precisely from log fields.
Guard 2 is approximated from Bovada's observed player list (can't reconstruct
order from the log alone), so we flag it as approximate.
"""
import json
from collections import defaultdict

with open("data/bets/bet_log.json") as f:
    log = json.load(f)

settled = [b for b in log if b.get("outcome") in ("W", "L")]

def get_direction(b):
    desc = b.get("description", "").upper()
    if "UNDER" in desc: return "UNDER"
    if "OVER" in desc: return "OVER"
    return ""

def winrate_str(bets):
    w = sum(1 for b in bets if b.get("outcome") == "W")
    l = sum(1 for b in bets if b.get("outcome") == "L")
    total = w + l
    if total == 0:
        return "0W 0L  —"
    return f"{w}W {l}L  {int(w/total*100)}%  (n={total})"

# ── Baseline breakdown ──────────────────────────────────────────────────────
print("=" * 65)
print("BASELINE — all settled bets")
print("=" * 65)
by_market = defaultdict(list)
for b in settled:
    m = b.get("market", "unknown")
    d = get_direction(b)
    key = f"{m}_{d}" if d else m
    by_market[key].append(b)

for k, bets in sorted(by_market.items(), key=lambda x: -len(x[1])):
    print(f"  {k:35s} {winrate_str(bets)}")

total_w = sum(1 for b in settled if b["outcome"] == "W")
total_l = len(settled) - total_w
print(f"\n  {'TOTAL':35s} {total_w}W {total_l}L  {int(total_w/len(settled)*100)}%  (n={len(settled)})")

# ── Guard 1: K OVER — drop if edge < 15% ───────────────────────────────────
print("\n" + "=" * 65)
print("GUARD 1: Block K OVERs with edge < 15%")
print("=" * 65)
k_overs = [b for b in settled if b.get("market") == "prop_pitcher_k" and get_direction(b) == "OVER"]
k_over_dropped = k_overs  # block ALL K OVERs
print(f"  K OVERs dropped (ALL): {len(k_over_dropped)}  {winrate_str(k_over_dropped)}")
print(f"  Net bets removed: {len(k_over_dropped)}")

# ── Guard 2: Runs OVER — drop spots 6-9 (known from log descriptions) ──────
print("\n" + "=" * 65)
print("GUARD 2: Block runs OVERs for lineup spots 6-9")
print("  (approximated: flagging Yastrzemski, Dubon and other confirmed")
print("   bottom-order players from the logged descriptions)")
print("=" * 65)
# Known bottom-order players from the bet log (manually confirmed from game lineups)
_bottom_order_keywords = [
    "yastrzemski", "dubon", "dominic smith", "leo rivas",
    "ceddanne rafaela", "marcelo mayer", "carlos narvaez",
    "jake fraley", "taylor walls", "ben williamson",
]
runs_overs = [b for b in settled if b.get("market") == "prop_runs" and get_direction(b) == "OVER"]
runs_over_dropped = [b for b in runs_overs
                     if any(kw in b.get("description", "").lower() for kw in _bottom_order_keywords)]
runs_over_kept = [b for b in runs_overs if b not in runs_over_dropped]
print(f"  Runs OVERs dropped (spot 6-9): {len(runs_over_dropped)}  {winrate_str(runs_over_dropped)}")
print(f"  Runs OVERs kept   (spot 1-5):  {len(runs_over_kept)}  {winrate_str(runs_over_kept)}")

# ── Guard 3: K UNDER low-line — drop when line <= 4.5 and big miss ─────────
print("\n" + "=" * 65)
print("GUARD 3: K UNDER low-line guard (line <= 4.5, proj gap < 1.0 K)")
print("  (proxy: drop bets where we cannot confirm >= 1 K gap;")
print("   measured by losses where actual exceeded line by >= 1)")
print("=" * 65)
k_unders_low = [b for b in settled
                if b.get("market") == "prop_pitcher_k"
                and get_direction(b) == "UNDER"
                and float(b.get("line", 99)) <= 4.5]
k_unders_low_kept = [b for b in k_unders_low if b.get("edge_pct", 0) >= 5.0]
# Among kept, show how many big-miss losses (actual >= line + 1)
big_miss_losses = [b for b in k_unders_low_kept
                   if b.get("outcome") == "L"
                   and b.get("actual") is not None
                   and float(b.get("actual", 0)) >= float(b.get("line", 0)) + 1.0]
print(f"  Low-line K UNDERs (line<=4.5): {len(k_unders_low)}  {winrate_str(k_unders_low)}")
print(f"  Big-miss losses (actual>=line+1): {len(big_miss_losses)}")
for b in big_miss_losses:
    print(f"    {b['description'][:55]:55s} line {b['line']} actual {b.get('actual')}")

# ── Pro-forma combined ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PRO-FORMA: Apply all guards to settled history")
print("=" * 65)
dropped_ids = set()
for b in k_over_dropped:
    dropped_ids.add(id(b))
for b in runs_over_dropped:
    dropped_ids.add(id(b))

filtered = [b for b in settled if id(b) not in dropped_ids]
fw = sum(1 for b in filtered if b["outcome"] == "W")
fl = len(filtered) - fw
print(f"  Bets removed: {len(settled) - len(filtered)}")
print(f"  Remaining:    {fw}W {fl}L  {int(fw/len(filtered)*100) if filtered else 0}%  (n={len(filtered)})")

# Re-breakdown after guards
print()
by_market2 = defaultdict(list)
for b in filtered:
    m = b.get("market", "unknown")
    d = get_direction(b)
    key = f"{m}_{d}" if d else m
    by_market2[key].append(b)
for k, bets in sorted(by_market2.items(), key=lambda x: -len(x[1])):
    print(f"  {k:35s} {winrate_str(bets)}")

# ── By date ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("BY DATE — filtered record")
print("=" * 65)
by_date = defaultdict(list)
for b in filtered:
    by_date[b.get("date", "?")].append(b)
for dt in sorted(by_date):
    bets = by_date[dt]
    w = sum(1 for b in bets if b["outcome"] == "W")
    l = len(bets) - w
    print(f"  {dt}  {w}W {l}L  {int(w/len(bets)*100) if bets else 0}%")
