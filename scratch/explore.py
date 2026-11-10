"""Scratch profiling of the raw human interactome (throwaway exploratory script)."""

import collections
import gzip
import time

import networkx as nx

t0 = time.time()
edges = []
with gzip.open("PP-Pathways_ppi.csv.gz", "rt") as f:
    raw = f.read()
for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
    line = line.strip()
    if not line:
        continue
    a, b = line.split(",")
    if a != b:
        edges.append((a, b))
print("raw edge rows:", len(edges), f"{time.time() - t0:.1f}s")

G = nx.Graph()
G.add_edges_from(edges)
G.remove_edges_from(nx.selfloop_edges(G))
print(
    "nodes:",
    G.number_of_nodes(),
    "edges:",
    G.number_of_edges(),
    f"density {nx.density(G):.3e}",
)

comps = sorted(nx.connected_components(G), key=len, reverse=True)
print("components:", len(comps), "largest:", len(comps[0]), "2nd:", len(comps[1]) if len(comps) > 1 else 0)
deg = dict(G.degree())
vals = sorted(deg.values())
n = len(vals)
print(f"mean degree {sum(vals) / n:.1f} median {vals[n // 2]} max {vals[-1]}")
print("degrees>=50:", sum(1 for v in vals if v >= 50), ">=100:", sum(1 for v in vals if v >= 100))

core = nx.core_number(G)
cd = collections.Counter(core.values())
print("core number distribution (top):", sorted(cd.items(), reverse=True)[:15])

s = sum(d * (d - 1) // 2 for d in vals)
print(f"sum C(deg,2) = {s:.2e}  (2-hop candidate upper bound)")

for k in (2, 3, 5, 8, 10, 15, 20):
    sub = [nd for nd, c in core.items() if c >= k]
    S = G.subgraph(sub)
    print(f"core>={k}: nodes {S.number_of_nodes()} edges {S.number_of_edges()}")

print(f"total {time.time() - t0:.1f}s")
