"""67-event lifecycle-phase distribution for tab:phasedist (paper).

Per event: dominant phase (plurality of bridge pairs), bridge concentration
(fraction in dominant phase), #bridges (len bridge_pairs). NO LLM/model calls.
"""
import json, glob
from collections import Counter
from pathlib import Path

PHASE_NORM = {"emergence":"Emergence","diffusion":"Diffusion",
              "peak":"Peak","decline":"Decline"}
ORDER = ["Decline","Emergence","Diffusion","Peak"]

dd = Path("data/cphot/processed/test_real")
rows = []
for f in sorted(glob.glob(str(dd/"*.json"))):
    if any(s in f for s in ("_s5","_s6","_emb")):
        continue
    ev = json.load(open(f))
    eid = ev["event_id"]
    nbr = len(ev.get("bridge_pairs", []))
    bp = [p for p in ev.get("scored_pairs", []) if p.get("label")==1]
    ph = Counter(PHASE_NORM.get(p.get("phase","").lower(), p.get("phase","?")) for p in bp)
    if not ph:
        rows.append((eid, "?", 0.0, nbr, 0)); continue
    dom, dn = ph.most_common(1)[0]
    conc = dn/sum(ph.values())
    rows.append((eid, dom, conc, nbr, sum(ph.values())))

# group by dominant phase, sort by conc desc within group
by = {ph:[] for ph in ORDER}
for r in rows:
    by.setdefault(r[1], []).append(r)
print(f"total events: {len(rows)}")
for ph in ORDER + [k for k in by if k not in ORDER]:
    grp = sorted(by.get(ph,[]), key=lambda r:-r[2])
    if not grp: continue
    print(f"\n=== {ph}-dominant ({len(grp)} events) ===")
    for eid,d,conc,nbr,scored in grp:
        print(f"{eid:34s} {ph:10s} {conc*100:5.0f}%  #br={nbr:4d} (scored_pos={scored})")
