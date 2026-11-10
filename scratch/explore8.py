"""Scratch: compare pruned spring vs Kamada-Kawai for the core-interactome figure."""

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

t0 = time.time()
edges = pd.read_csv("data/ppi_edges_clean.csv", dtype=str)
G = nx.Graph()
G.add_edges_from(zip(edges["source"], edges["target"], strict=False))
G = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0]).copy()
sym = dict(zip(edges["source"], edges["source_symbol"], strict=False))
sym.update(dict(zip(edges["target"], edges["target_symbol"], strict=False)))

comms = nx.community.louvain_communities(G, seed=42, resolution=3.0)
community_of = {n: i for i, members in enumerate(comms) for n in members}

top_nodes = [n for n, _ in sorted(G.degree, key=lambda kv: kv[1], reverse=True)[:600]]
CORE = G.subgraph(top_nodes).copy()
CORE.remove_nodes_from([n for n in CORE if CORE.degree(n) < 2])
CORE = CORE.subgraph(sorted(nx.connected_components(CORE), key=len, reverse=True)[0]).copy()
print(f"core: {CORE.number_of_nodes()} nodes / {CORE.number_of_edges()} edges ({time.time() - t0:.1f}s)")

node_deg = np.array([CORE.degree(n) for n in CORE])
node_comm = np.array([community_of.get(n, 0) for n in CORE])
palette = plt.get_cmap("tab20")


def draw(pos, name, title):
    fig, ax = plt.subplots(figsize=(11, 10))
    nx.draw_networkx_edges(CORE, pos, alpha=0.05, width=0.45, ax=ax)
    nx.draw_networkx_nodes(
        CORE, pos,
        node_size=14 + 150 * (node_deg / node_deg.max()) ** 1.5,
        node_color=[palette(c % 20) for c in node_comm],
        linewidths=0.4, edgecolors="white", alpha=0.92, ax=ax,
    )
    top = sorted(CORE, key=lambda x: -CORE.degree(x))[:18]
    nx.draw_networkx_labels(CORE, pos, labels={n: sym.get(n, n) for n in top},
                            font_size=9, font_weight="bold", ax=ax,
                            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(name, dpi=110, bbox_inches="tight")
    plt.close(fig)


t = time.time()
draw(nx.spring_layout(CORE, seed=42, k=0.32, iterations=60),
     "scratch_A_spring.png", "A: spring layout (pruned core)")
print(f"spring {time.time() - t:.1f}s")

t = time.time()
draw(nx.kamada_kawai_layout(CORE),
     "scratch_B_kk.png", "B: Kamada-Kawai layout (pruned core)")
print(f"kk {time.time() - t:.1f}s")
print(f"total {time.time() - t0:.1f}s")
