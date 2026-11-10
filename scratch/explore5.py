"""Scratch: inspect real DDR-module top predictions before writing the notebook."""

import gzip
import math
import random

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

id2sym, id2desc, sym2id = {}, {}, {}
with gzip.open("Homo_sapiens.gene_info.gz", "rt") as f:
    next(f)
    rows = [line.rstrip("\n").split("\t") for line in f]
for p in rows:
    if len(p) < 11:
        continue
    id2sym[p[1]] = p[2]
    id2desc[p[1]] = p[8]
    sym2id[p[2]] = p[1]
for p in rows:
    if len(p) < 11:
        continue
    for syn in p[4].split("|"):
        if syn and syn != "-":
            sym2id.setdefault(syn, p[1])

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
DDR = list(dict.fromkeys(DDR))

rng = random.Random(42)
edge_list = list(G.edges())
rng.shuffle(edge_list)
n_test = int(0.2 * len(edge_list))
test_set = {tuple(sorted(e)) for e in edge_list[:n_test]}
G_train = nx.Graph()
G_train.add_edges_from(edge_list[n_test:])
adj = {n: set(nb) for n, nb in G_train.adjacency()}
deg = dict(G_train.degree())

seeds = [sym2id[s] for s in DDR if s in sym2id and sym2id[s] in G_train]
print("seeds:", len(seeds))
seed_set = set(seeds)
expand = set(seed_set)
for s in seeds:
    expand.update(G_train.neighbors(s))
core = set()
for v in expand:
    if v in seed_set or sum(1 for s in seed_set if G_train.has_edge(v, s)) >= 5:
        core.add(v)
MOD = G_train.subgraph(core).copy()
print(f"module: {MOD.number_of_nodes()} nodes / {MOD.number_of_edges()} edges")

nodes = sorted(core)
cands = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:] if not G.has_edge(u, v)]
print("candidates:", len(cands))

aa, cn, pa = [], [], []
for u, v in cands:
    inter = adj.get(u, set()) & adj.get(v, set())
    cn.append(len(inter))
    aa.append(sum(1.0 / math.log(deg[w]) for w in inter if deg[w] > 1))
    pa.append(deg[u] * deg[v])
aa = np.array(aa)
cn = np.array(cn)
pa = np.array(pa)

def sym(n):
    return id2sym.get(n, n)
for rank_i in np.argsort(-aa)[:20]:
    u, v = cands[rank_i]
    inter = [sym(w) for w in (adj[u] & adj[v])][:4]
    print(f"AA {aa[rank_i]:6.2f} CN {cn[rank_i]:2d} {sym(u):8s}({deg[u]:4d}) - {sym(v):8s}({deg[v]:4d}) hub={max(deg[u], deg[v]) > 400} via {inter}")

print("\nholdout check: top-1000 AA hits in test_set =", sum(1 for i in np.argsort(-aa)[:1000] if cands[i] in test_set))
print("total test edges inside module candidates =", sum(1 for c in cands if c in test_set))
print("\nhub-filtered top-15 (max deg <= 400):")
n = 0
for rank_i in np.argsort(-aa):
    u, v = cands[rank_i]
    if max(deg[u], deg[v]) > 400:
        continue
    inter = [sym(w) for w in (adj[u] & adj[v])][:4]
    print(f"  AA {aa[rank_i]:6.2f} CN {cn[rank_i]:2d} {sym(u):8s}({deg[u]:3d}) - {sym(v):8s}({deg[v]:3d}) via {inter}")
    n += 1
    if n >= 15:
        break

# --- module-local evaluation: candidates = non-edges of G_train inside module
nodes_g = set(core)
all_pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]]
train_pairs = set()
for u in nodes:
    for v in adj[u]:
        if v in nodes_g and u < v:
            train_pairs.add((u, v))
cand_train = [p for p in all_pairs if p not in train_pairs]
print(f"\nmodule non-edges of G_train: {len(cand_train)}")
test_in_mod = [p for p in cand_train if p in test_set]
print("held-out edges inside module:", len(test_in_mod))
aa2 = []
for u, v in cand_train:
    inter = adj[u] & adj[v]
    aa2.append(sum(1.0 / math.log(deg[w]) for w in inter if deg[w] > 1))
aa2 = np.array(aa2)
order = np.argsort(-aa2)
tset = set(test_in_mod)
for K in (100, 500, 1000, 5000):
    hits = sum(1 for i in order[:K] if cand_train[i] in tset)
    chance = K * len(test_in_mod) / len(cand_train)
    print(f"  K={K:5d} hits={hits:3d} recall={hits / max(1, len(test_in_mod)):.3f} enrichment={hits / chance if chance else 0:.1f}x")
