"""Scratch: prototype the community-aware layout for the network figures."""

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

t0 = time.time()
edges = pd.read_csv("data/ppi_edges_clean.csv", dtype=str)
G = nx.Graph()
G.add_edges_from(zip(edges["source"], edges["target"], strict=False))
comps = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(comps[0]).copy()

sym = dict(zip(edges["source"], edges["source_symbol"], strict=False))
sym.update(dict(zip(edges["target"], edges["target_symbol"], strict=False)))
print(f"loaded {G.number_of_nodes()} / {G.number_of_edges()} in {time.time() - t0:.1f}s")


def community_layout(graph, seed=42, k=None, iterations=100, spread=0.08):
    """Circle-initialised spring layout so communities stay visually grouped."""
    prng = np.random.default_rng(seed)
    groups = {}
    for node, cid in nx.get_node_attributes(graph, "community").items():
        groups.setdefault(cid, []).append(node)
    keys = sorted(groups)
    positions = {}
    angles = np.linspace(0, 2 * np.pi, max(len(keys), 1), endpoint=False)
    for angle, cid in zip(angles, keys, strict=False):
        centre = np.array([np.cos(angle), np.sin(angle)])
        for node in groups[cid]:
            positions[node] = centre + prng.normal(0, spread, 2)
    for node in graph:
        positions.setdefault(node, prng.normal(0, spread, 2))
    return nx.spring_layout(graph, pos=positions, k=k, iterations=iterations, seed=seed)


CORE = G.subgraph([n for n, _ in sorted(G.degree, key=lambda kv: kv[1], reverse=True)[:600]]).copy()
comms = nx.community.louvain_communities(G, seed=42, resolution=3.0)
community_of = {n: i for i, members in enumerate(comms) for n in members}
nx.set_node_attributes(CORE, {n: community_of.get(n, -1) for n in CORE}, "community")
print(f"core prepared in {time.time() - t0:.1f}s; communities {len(set(community_of.values()))}")

pos = community_layout(CORE, k=0.45, iterations=90)
node_deg = np.array([CORE.degree(n) for n in CORE])
node_comm = np.array([community_of.get(n, 0) for n in CORE])
palette = plt.get_cmap("tab20")

fig, ax = plt.subplots(figsize=(11, 10))
nx.draw_networkx_edges(CORE, pos, alpha=0.05, width=0.45, ax=ax)
nx.draw_networkx_nodes(
    CORE, pos,
    node_size=14 + 150 * (node_deg / node_deg.max()) ** 1.5,
    node_color=[palette(c % 20) for c in node_comm],
    linewidths=0.4, edgecolors="white", alpha=0.92, ax=ax,
)
top = sorted(CORE, key=lambda x: -CORE.degree(x))[:20]
labels = {n: sym.get(n, n) for n in top}
nx.draw_networkx_labels(CORE, pos, labels=labels, font_size=9, font_weight="bold", ax=ax,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75})
ax.set_title("Core interactome: 600 highest-degree proteins\nnode size = degree, colour = fine Louvain community", fontsize=12)
ax.axis("off")
fig.tight_layout()
fig.savefig("scratch_core_layout.png", dpi=110, bbox_inches="tight")
print(f"done in {time.time() - t0:.1f}s -> scratch_core_layout.png")
