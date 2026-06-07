"""Compute ACHIEVABLE MABD headroom per event and per K.

A MABD-zone (or rescued DISCARD) true bridge can only improve AP@K if the
event has fewer than K true bridges already correctly placed in the BRIDGE
zone. Otherwise the extra bridge ranks below the top-K and changes nothing.

Reads logs/mabd_routing_analysis.json produced by analyze_mabd_routing.py.
"""
import json
from pathlib import Path

data = json.loads(Path("logs/mabd_routing_analysis.json").read_text())
per_event = data["per_event"]

print("Events with ACHIEVABLE MABD/rescue headroom (BRIDGE-zone bridges < K):\n")
for K in (5, 20, 50):
    print(f"=== K={K} ===")
    affected = []
    for ev in per_event:
        in_bridge = ev["in_BRIDGE"]
        in_mabd = ev["in_MABD"]
        # cosine-rescuable DISCARD bridges (improvement E)
        rescuable = sum(1 for d in ev["discard_bridge_details"] if d["s1"] >= 0.85)
        # headroom in top-K not already filled by BRIDGE-zone bridges
        slack = max(0, K - in_bridge)
        gain_mabd = min(slack, in_mabd)
        gain_rescue = min(slack, in_mabd + rescuable)
        if slack > 0 and (in_mabd > 0 or rescuable > 0):
            affected.append((ev["event_id"], ev["n_bridge_nodes"], in_bridge,
                             in_mabd, rescuable, gain_mabd, gain_rescue))
    if not affected:
        print("  (none)\n")
        continue
    print(f"  {'event':40s} {'tot':>4} {'BR':>3} {'MABD':>4} {'resc':>4} "
          f"{'D_gain':>6} {'E_gain':>6}")
    tot_d = tot_e = 0
    for eid, tot, br, mabd, resc, gd, ge in affected:
        print(f"  {eid:40s} {tot:4d} {br:3d} {mabd:4d} {resc:4d} {gd:6d} {ge:6d}")
        tot_d += gd
        tot_e += ge
    print(f"  {'TOTAL achievable bridge-node gains':40s} "
          f"{'':4} {'':3} {'':4} {'':4} {tot_d:6d} {tot_e:6d}\n")
