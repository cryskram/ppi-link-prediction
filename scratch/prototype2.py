"""Scratch: test degree-matched negatives to expose the PA degree confound."""

import gzip
import math
import random
import time

import networkx as nx
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

t0 = time.time()
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

rng = random.Random(42)
edge_list = list(G.edges())
rng.shuffle(edge_list)
n_test = int(0.2 * len(edge_list))
test_edges = [tuple(sorted(e)) for e in edge_list[:n_test]]
train_edges = [tuple(sorted(e)) for e in edge_list[n_test:]]
G_train = nx.Graph()
G_train.add_edges_from(train_edges)
adj = {n: set(nb) for n, nb in G_train.adjacency()}
deg = dict(G_train.degree())
all_edges = {tuple(sorted(e)) for e in G.edges()}
nodes = np.array(list(G_train.nodes()))
idx = {n: i for i, n in enumerate(nodes)}
dvals = np.array([deg.get(n, 0) for n in nodes], dtype=float)

RATIO = 5
n_neg = RATIO * len(test_edges)
test_set = set(test_edges)

# --- degree-matched sampling via a pre-sampled pool
t = time.time()
POOL = 6_000_000
prng = np.random.default_rng(42)
iu = prng.integers(0, len(nodes), POOL)
iv = prng.integers(0, len(nodes), POOL)
keep = iu != iv
iu, iv = iu[keep], iv[keep]
prod = dvals[iu] * dvals[iv]
logp = np.log10(prod + 1.0)
print(f"pool built {len(iu)} in {time.time() - t:.1f}s")

pos_logp = np.array([math.log10(deg.get(u, 0) * deg.get(v, 0) + 1.0) for u, v in test_edges])
bins = np.linspace(logp.min(), logp.max(), 41)
pos_bin = np.digitize(pos_logp, bins)
pool_bin = np.digitize(logp, bins)

order = np.argsort(pool_bin)
sorted_bin = pool_bin[order]
starts = np.searchsorted(sorted_bin, np.arange(bins.size + 2))

picked = []
need = np.bincount(pos_bin, minlength=bins.size + 2)
for b in np.nonzero(need)[0]:
    lo, hi = starts[b], starts[b + 1]
    if hi - lo < need[b]:
        continue
    sel = prng.choice(np.arange(lo, hi), size=need[b], replace=False)
    rows = order[sel]
    picked.append(np.column_stack([iu[rows], iv[rows]]))
cand = np.vstack(picked)
print(f"degree-matched candidates {len(cand)} in {time.time() - t:.1f}s")

dm_neg = []
seen = set()
for a, b in cand:
    u, v = nodes[a], nodes[b]
    p = tuple(sorted((u, v)))
    if p in all_edges or p in seen:
        continue
    seen.add(p)
    dm_neg.append(p)
print(f"degree-matched negatives {len(dm_neg)} (unique, non-edge)")


def score_pairs(pairs):
    cn, jac, aa, pa = [], [], [], []
    for u, v in pairs:
        nu, nv = adj.get(u, set()), adj.get(v, set())
        inter = nu & nv
        c = len(inter)
        union = len(nu) + len(nv) - c
        cn.append(c)
        jac.append(c / union if union else 0.0)
        aa.append(sum(1.0 / math.log(deg[w]) for w in inter if deg[w] > 1))
        pa.append(deg.get(u, 0) * deg.get(v, 0))
    return {"CN": np.array(cn, float), "Jaccard": np.array(jac, float),
            "AdamicAdar": np.array(aa, float), "PA": np.array(pa, float)}


pairs = list(test_edges) + dm_neg
rng.shuffle(pairs)
y = np.array([1 if p in test_set else 0 for p in pairs])
sc = score_pairs(pairs)
print(f"\n--- DEGREE-MATCHED ({len(pairs)} pairs) baseline precision {y.mean():.3f} ---")
for algo in ("CN", "Jaccard", "AdamicAdar", "PA"):
    s = sc[algo]
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    o = np.argsort(-s, kind="stable")
    ys = y[o]
    line = f"{algo:11s} AUC {auc:.4f} AP {ap:.4f} |"
    for k in (100, 1000, 5000, 20000):
        tp = ys[:k].sum()
        p = tp / k
        r = tp / y.sum()
        f1 = 2 * p * r / (p + r) if p + r else 0
        line += f" P@{k} {p:.3f} F1 {f1:.3f} |"
    print(line)
print(f"total {time.time() - t0:.1f}s")
