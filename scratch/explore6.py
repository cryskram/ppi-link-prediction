"""Scratch: proper power-law MLE and a Louvain resolution sweep."""

import gzip

import networkx as nx
import numpy as np

edges = []
with gzip.open("PP-Pathways_ppi.csv.gz", "rt") as f:
    raw = f.read()
for line in raw.replace("\r", "\n").split("\n"):
    line = line.strip()
    if not line:
        continue
    a, b = line.split(",")
    if a != b:
        edges.append((a, b))
G = nx.Graph()
G.add_edges_from(edges)
comps = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(comps[0]).copy()

id2sym = {}
with gzip.open("Homo_sapiens.gene_info.gz", "rt") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) > 2:
            id2sym.setdefault(p[1], p[2])


def sym(n):
    return id2sym.get(n, n)


deg = np.array([d for _, d in G.degree()])

# --- CCDF slope -> gamma = 1 + slope
s = np.sort(deg)[::-1]
ccdf = np.arange(1, len(s) + 1) / len(s)
tail = s >= 3
slope, intercept = np.polyfit(np.log10(s[tail]), np.log10(ccdf[tail]), 1)
print(f"CCDF slope {slope:.3f} -> gamma ~ {1 - slope:.2f}")

# --- discrete MLE (Clauset) for several k_min
for kmin in (5, 10, 20, 50):
    tail_deg = deg[deg >= kmin]
    gamma_mle = 1 + len(tail_deg) / np.sum(np.log(tail_deg / (kmin - 0.5)))
    print(f"MLE k_min={kmin:3d}: n_tail={len(tail_deg):5d} gamma={gamma_mle:.2f}")

# --- Louvain resolution sweep
for res in (1.0, 1.5, 2.0, 3.0):
    comms = nx.community.louvain_communities(G, seed=42, resolution=res)
    q = nx.community.modularity(G, comms)
    print(f"\nresolution={res}: {len(comms)} communities, Q={q:.3f}")
    for members in sorted(comms, key=len, reverse=True)[:6]:
        hubs = [sym(n) for n, _ in sorted(((n, G.degree(n)) for n in members), key=lambda x: -x[1])[:5]]
        dense = 2 * G.subgraph(members).number_of_edges() / (len(members) * (len(members) - 1))
        print(f"   size {len(members):5d} density {dense:.3f} -> {', '.join(hubs)}")
