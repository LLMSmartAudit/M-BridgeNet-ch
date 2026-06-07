"""Weak-tie / sole-connector test: are MABD-overlooked bridges 'crucial'?

Granovetter weak tie: an edge whose removal disconnects two otherwise-separated
communities. Graph-theoretically, that is a CUT EDGE (a.k.a. graph bridge) — an
edge not contained in any cycle.

For each event we build the undirected ground-truth bridge graph
  nodes = posts appearing in any annotated bridge pair
  edges = annotated bridge pairs (cross-platform narrative transfers)
and compute the set of cut edges (nx.bridges). A true-bridge SOURCE NODE is a
'sole connector' if any of its incident bridge edges is a cut edge — i.e. it is
the only carrier linking its two post-communities.

We then ask: are MABD-zone (overlooked) source nodes MORE often sole connectors
than direct-BRIDGE source nodes? If yes, 'crucial' is earned.

We also report route redundancy: the number of OTHER true-bridge source nodes
sharing the same undirected platform-pair route in the event (lower = more
crucial, fewer alternative carriers).

NO LLM calls.
"""
from __future__ import annotations
import argparse
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

import networkx as nx

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.schemas import Route

logging.getLogger().setLevel(logging.ERROR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--low-s2-simonly", type=float, default=0.0, dest="low_s2")
    ap.add_argument("--out", default="logs/mabd_weaktie_analysis.json")
    args = ap.parse_args()

    cfg = load_config(None)
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint=args.checkpoint, llm_client=None)
    ds = CPHotDataset(Path(args.data))
    data_dir = Path(args.data)

    # per-zone tallies
    z_stats = {z: {"n": 0, "sole": 0, "redundancy": []}
               for z in ("BRIDGE", "MABD", "DISCARD")}

    for event in ds:
        eid = event.get("event_id", "")
        bp = event["bridge_pairs"]
        if not bp:
            continue
        posts = event["posts"]
        plat = {p.post_id: p.platform for p in posts}

        # 1) Build GT bridge graph and find cut edges (weak ties)
        G = nx.Graph()
        for a, b in bp:
            G.add_edge(a, b)
        cut_edges = set()
        for u, v in nx.bridges(G):           # graph-theoretic bridges = cut edges
            cut_edges.add(frozenset((u, v)))

        # incident cut-edge lookup per source node
        sole_nodes = set()
        for a, b in bp:
            if frozenset((a, b)) in cut_edges:
                sole_nodes.add(a)            # source node a is a sole connector

        # 2) Route redundancy: per source node, count co-route true-bridge sources
        route_sources = defaultdict(set)     # undirected platform pair -> {source ids}
        node_route = {}
        for a, b in bp:
            r = tuple(sorted((plat.get(a, "?"), plat.get(b, "?"))))
            route_sources[r].add(a)
            node_route[a] = r

        # 3) Determine routing zone per source node via Stage 1+2
        gt_sources = {a for a, _ in bp}
        s5_map = {}
        p5 = data_dir / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        cands = pipe.run_stage1_stage2(
            posts, event["hourly_volumes"],
            low_s2_simonly_threshold=args.low_s2,
            s5_map=s5_map if s5_map else None)
        best = {}
        for c in cands:
            if c.post_a.post_id in gt_sources:
                pid = c.post_a.post_id
                if pid not in best or c.score_S > best[pid].score_S:
                    best[pid] = c

        for node in gt_sources:
            if node not in best:
                continue
            c = best[node]
            z = ("BRIDGE" if c.route == Route.BRIDGE
                 else "MABD" if c.route == Route.MABD else "DISCARD")
            z_stats[z]["n"] += 1
            if node in sole_nodes:
                z_stats[z]["sole"] += 1
            # redundancy: other sources on same route
            r = node_route.get(node)
            red = len(route_sources[r]) - 1 if r else 0
            z_stats[z]["redundancy"].append(red)

    print("=" * 74)
    print("WEAK-TIE / SOLE-CONNECTOR TEST (67-event test_real)")
    print("  sole-connector = node's bridge edge is a graph cut edge")
    print("  route redundancy = # OTHER true-bridge sources on the same platform route")
    print("=" * 74)
    print(f"\n  {'zone':8s} {'n':>5} {'sole-conn %':>12} {'median redundancy':>18}")
    for z in ("BRIDGE", "MABD", "DISCARD"):
        s = z_stats[z]
        pct = 100 * s["sole"] / s["n"] if s["n"] else 0.0
        med = st.median(s["redundancy"]) if s["redundancy"] else 0
        print(f"  {z:8s} {s['n']:>5} {pct:>11.1f}% {med:>18.1f}")

    # decisive contrast
    def pct(z):
        s = z_stats[z]
        return 100 * s["sole"] / s["n"] if s["n"] else 0.0
    print("\n" + "=" * 74)
    print("DECISIVE CONTRAST (MABD-overlooked vs direct-BRIDGE)")
    print("=" * 74)
    d = pct("MABD") - pct("BRIDGE")
    print(f"  Δ sole-connector rate = {d:+.1f} pp   "
          f"{'✓ overlooked bridges ARE more often sole connectors (crucial)' if d > 0 else '✗ NOT more often sole connectors'}")
    rb = st.median(z_stats['BRIDGE']['redundancy']) if z_stats['BRIDGE']['redundancy'] else 0
    rm = st.median(z_stats['MABD']['redundancy']) if z_stats['MABD']['redundancy'] else 0
    print(f"  median redundancy: BRIDGE={rb:.1f}  MABD={rm:.1f}   "
          f"{'✓ overlooked have fewer alternative carriers' if rm < rb else '✗ not fewer'}")

    Path(args.out).write_text(json.dumps(
        {z: {"n": s["n"], "sole": s["sole"],
             "sole_pct": (100*s["sole"]/s["n"] if s["n"] else 0),
             "median_redundancy": (st.median(s["redundancy"]) if s["redundancy"] else 0)}
         for z, s in z_stats.items()}, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
