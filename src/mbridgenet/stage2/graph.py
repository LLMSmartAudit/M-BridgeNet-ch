from __future__ import annotations
from typing import Dict, List, Tuple
import networkx as nx
from mbridgenet.schemas import Post


def build_event_graph(
    posts: List[Post],
    candidate_pairs: List[Tuple[str, str]],
    same_platform_window_hours: float = 24.0,
) -> nx.Graph:
    """Build G_E = (V, E_same ∪ E_cross).

    E_same: edges between posts on the same platform within
            same_platform_window_hours of each other (chronological).
    E_cross: edges from candidate bridge pairs.
    Nodes carry 'platform' and 'account_id' attributes.
    """
    G = nx.Graph()

    # Add nodes
    for p in posts:
        G.add_node(p.post_id, platform=p.platform,
                   account_id=p.account_id, timestamp=p.timestamp)

    # E_same: connect consecutive posts on same platform within window
    by_platform: Dict[str, List[Post]] = {}
    for p in posts:
        by_platform.setdefault(p.platform, []).append(p)

    for platform_posts in by_platform.values():
        sorted_posts = sorted(platform_posts, key=lambda p: p.timestamp)
        for i in range(len(sorted_posts) - 1):
            pa, pb = sorted_posts[i], sorted_posts[i + 1]
            gap_h = (pb.timestamp - pa.timestamp).total_seconds() / 3600
            if gap_h <= same_platform_window_hours:
                G.add_edge(pa.post_id, pb.post_id, edge_type="same")

    # E_cross: candidate bridge pairs
    post_ids = {p.post_id for p in posts}
    for (id_a, id_b) in candidate_pairs:
        if id_a in post_ids and id_b in post_ids:
            G.add_edge(id_a, id_b, edge_type="cross")

    return G


def compute_betweenness(G: nx.Graph, k_approx: int = 200) -> Dict[str, float]:
    """Compute betweenness centrality for all nodes in G.

    For large graphs (>500 nodes) uses approximate sampling with k_approx pivots
    to keep runtime tractable.
    """
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_nodes() > 500:
        k = min(k_approx, G.number_of_nodes())
        return nx.betweenness_centrality(G, k=k, normalized=True)
    return nx.betweenness_centrality(G, normalized=True)
