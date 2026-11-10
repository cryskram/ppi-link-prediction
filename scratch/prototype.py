"""Scratch prototype: full link-prediction pipeline to validate design + timing."""

import gzip
import math
import random
import time

import networkx as nx
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

t0 = time.time()

# ---------- load
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
print(f"loaded {G.number_of_nodes()} nodes / {G.number_of_edges()} edges  ({time.time() - t0:.1f}s)")

anno = {}
sym2id = {}
with gzip.open("Homo_sapiens.gene_info.gz", "rt") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 11:
            continue
        anno[p[1]] = {"symbol": p[2], "chromosome": p[6], "description": p[8]}
        sym2id.setdefault(p[2], p[1])

# ---------- 80/20 split
rng = random.Random(42)
edge_list = list(G.edges())
rng.shuffle(edge_list)
n_test = int(0.2 * len(edge_list))
test_edges = [tuple(sorted(e)) for e in edge_list[:n_test]]
train_edges = [tuple(sorted(e)) for e in edge_list[n_test:]]
G_train = nx.Graph()
G_train.add_edges_from(train_edges)
print(f"train {G_train.number_of_edges()} | test {len(test_edges)}")

# ---------- negatives
all_edges = {tuple(sorted(e)) for e in G.edges()}
nodes = list(G.nodes())
RATIO = 5


def sample_easy(n):
    out = set()
    while len(out) < n:
        u, v = rng.sample(nodes, 2)
        p = tuple(sorted((u, v)))
        if p not in all_edges:
            out.add(p)
    return list(out)


easy_neg = sample_easy(RATIO * len(test_edges))

# hard negatives: distance-2 non-edges in train graph
adj = {n: set(nb) for n, nb in G_train.adjacency()}
hard_neg = set()
tries = 0
while len(hard_neg) < RATIO * len(test_edges) and tries < 200 * RATIO * len(test_edges):
    tries += 1
    u = rng.choice(nodes)
    nbrs = list(adj.get(u, ()))
    if len(nbrs) < 2:
        continue
    a, b = rng.sample(nbrs, 2)
    p = tuple(sorted((a, b)))
    if p in all_edges or p in hard_neg:
        continue
    hard_neg.add(p)
hard_neg = list(hard_neg)
print(f"negatives: easy {len(easy_neg)} hard {len(hard_neg)} (tries {tries})  {time.time() - t0:.1f}s")

# ---------- scoring
deg = dict(G_train.degree())


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
    return {
        "CN": np.array(cn, dtype=float),
        "Jaccard": np.array(jac, dtype=float),
        "AdamicAdar": np.array(aa, dtype=float),
        "PA": np.array(pa, dtype=float),
    }


for name, negs in (("RANDOM", easy_neg), ("2-HOP HARD", hard_neg)):
    t = time.time()
    test_set = set(test_edges)
    pairs = list(test_edges) + list(negs)
    rng.shuffle(pairs)
    y = np.array([1 if p in test_set else 0 for p in pairs])
    sc = score_pairs(pairs)
    print(f"\n--- {name} candidates ({len(pairs)} pairs, {time.time() - t:.1f}s) ---")
    print(f"random baseline precision = {len(test_edges) / len(pairs):.3f}  positives in set = {y.sum()}")
    for algo in ("CN", "Jaccard", "AdamicAdar", "PA"):
        s = sc[algo]
        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        order = np.argsort(-s, kind="stable")
        ys = y[order]
        line = f"{algo:11s} AUC {auc:.4f} AP {ap:.4f} |"
        for k in (10, 100, 1000, 5000, 20000):
            tp = ys[:k].sum()
            p = tp / k
            r = tp / len(test_edges)
            f1 = 2 * p * r / (p + r) if p + r else 0
            line += f" P@{k} {p:.3f} R@{k} {r:.4f} F1 {f1:.3f} |"
        print(line)

# ---------- verify against networkx on a sample
sample = [(u, v) for u, v in test_edges[:200]]
mine = score_pairs(sample)
nx_cn = {(u, v): s for u, v, s in nx.common_neighbor_centrality(G_train, sample, alpha=0.0)}
nx_j = {(u, v): s for u, v, s in nx.jaccard_coefficient(G_train, sample)}
nx_aa = {(u, v): s for u, v, s in nx.adamic_adar_index(G_train, sample)}
nx_pa = {(u, v): s for u, v, s in nx.preferential_attachment(G_train, sample)}
print("\nverification vs networkx:")
print(" CN  max diff", max(abs(mine["CN"][i] - nx_cn[(u, v)]) for i, (u, v) in enumerate(sample)))
print(" JAC max diff", max(abs(mine["Jaccard"][i] - nx_j[(u, v)]) for i, (u, v) in enumerate(sample)))
print(" AA  max diff", max(abs(mine["AdamicAdar"][i] - nx_aa[(u, v)]) for i, (u, v) in enumerate(sample)))
print(" PA  max diff", max(abs(mine["PA"][i] - nx_pa[(u, v)]) for i, (u, v) in enumerate(sample)))

# ---------- DDR module
DDR = [
    "ATM", "ATR", "ATRIP", "BRCA1", "BRCA2", "PALB2", "RAD51", "RAD51B", "RAD51C", "RAD51D",
    "RAD52", "RAD54L", "BARD1", "BRIP1", "CHEK1", "CHEK2", "TP53", "TP53BP1", "MDC1", "H2AFX",
    "MRE11", "RAD50", "NBN", "RBBP8", "PARP1", "PARP2", "PARP3", "FANCA", "FANCB", "FANCC",
    "FANCD2", "FANCE", "FANCF", "FANCG", "FANCI", "FANCL", "FANCM", "MLH1", "MSH2", "MSH6",
    "PMS2", "BLM", "WRN", "ERCC1", "ERCC2", "ERCC3", "ERCC4", "ERCC5", "ERCC6", "ERCC8",
    "XPA", "XPC", "DDB1", "DDB2", "LIG1", "LIG3", "LIG4", "POLH", "PCNA", "REV1", "POLQ",
    "PRKDC", "XRCC1", "XRCC2", "XRCC3", "XRCC4", "XRCC5", "XRCC6", "RAD23A", "RAD23B",
    "CUL4A", "CUL4B", "MBD4", "TDG", "UNG", "APEX1", "OGG1", "MUTYH", "KAT5", "SETD8",
    "H2AFY", "DEK", "CHD1L", "DNA2", "GEN1", "MUS81", "EME1", "SLX4", "RTEL1", "TERF2",
    "POT1", "RAD9A", "RAD1", "HUS1", "CLSPN", "TOPBP1", "H2AFZ", "CDT1", "CDC25A", "WEE1",
    "PLK1", "AURKA", "AURKB", "BUB1", "TTK", "SMC1A", "SMC3", "STAG1", "STAG2", "ESCO1",
    "ESCO2", "SMC5", "SMC6", "NSMCE2", "PRPF19", "INO80", "RUVBL1", "RUVBL2", "ACTR5",
    "RAD18", "SHPRH", "HLTF", "UBE2N", "RNF8", "RNF168", "RNF4", "UBE2V2", "BMI1", "CBX4",
    "CBX8", "SMARCAD1", "BRCC3", "BABAM1",
]
seed_ids = [sym2id[s] for s in DDR if s in sym2id and sym2id[s] in G_train]
print(f"\nDDR seeds in train graph: {len(seed_ids)}/{len(DDR)}")

seed_set = set(seed_ids)
expand = set(seed_set)
for s in seed_ids:
    expand.update(G_train.neighbors(s))
keep = set(seed_set)
for v in expand - seed_set:
    if sum(1 for s in seed_set if G_train.has_edge(v, s)) >= 2:
        keep.add(v)
sub = G_train.subgraph(keep)
print(f"DDR module: {sub.number_of_nodes()} nodes / {sub.number_of_edges()} edges")

# exhaustive non-edge candidate pairs within module
mod_pairs = [
    (u, v) for i, u in enumerate(sorted(keep)) for v in sorted(keep)[i + 1:] if not G_train.has_edge(u, v)
]
print(f"module candidate non-edge pairs: {len(mod_pairs)}")
t = time.time()
mod_sc = score_pairs(mod_pairs)
print(f"scored in {time.time() - t:.1f}s")
for algo in ("CN", "Jaccard", "AdamicAdar", "PA"):
    idx = np.argsort(-mod_sc[algo])[:10]
    top = [(anno.get(mod_pairs[i][0], {}).get("symbol", mod_pairs[i][0]),
            anno.get(mod_pairs[i][1], {}).get("symbol", mod_pairs[i][1]),
            round(float(mod_sc[algo][i]), 3)) for i in idx[:5]]
    print(f" {algo:11s}", top)

print(f"\ntotal {time.time() - t0:.1f}s")
