"""Scratch: test DDR module extraction, annotations, and measure timings."""

import gzip
import time

import networkx as nx

t0 = time.time()

# --- load edges
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
print(f"graph {G.number_of_nodes()} nodes / {G.number_of_edges()} edges in {time.time() - t0:.1f}s")

# --- entrez -> symbol/description
anno = {}
with gzip.open("Homo_sapiens.gene_info.gz", "rt") as f:
    header = f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 11:
            continue
        anno[p[1]] = {"symbol": p[2], "chromosome": p[6], "description": p[8], "type": p[9]}

covered = sum(1 for n in G if n in anno)
print(f"annotation coverage: {covered}/{G.number_of_nodes()} = {covered / G.number_of_nodes():.1%}")

sym2id = {v["symbol"]: k for k, v in anno.items()}
print("unique symbols:", len(sym2id))

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

seed_ids = [sym2id[s] for s in DDR if s in sym2id and s in G]
print(f"DDR seeds present: {len(seed_ids)}/{len(DDR)}")
missing = [s for s in DDR if s not in sym2id]
print("missing symbols:", missing[:20])

# 1-hop neighbourhood
neigh = set(seed_ids)
for s in seed_ids:
    neigh.update(G.neighbors(s))
print(f"1-hop neighbourhood: {len(neigh)} nodes / {G.subgraph(neigh).number_of_edges()} edges")

# cohesion filter: keep neighbours touching >=2 seeds
seed_set = set(seed_ids)
keep = set(seed_set)
for v in neigh - seed_set:
    hits = sum(1 for s in seed_set if G.has_edge(v, s))
    if hits >= 2:
        keep.add(v)
print(f"cohesion>=2: {len(keep)} nodes / {G.subgraph(keep).number_of_edges()} edges")

# --- timings of measures on full graph
t = time.time()
nx.clustering(G)
print(f"clustering: {time.time() - t:.1f}s")

t = time.time()
nx.pagerank(G)
print(f"pagerank: {time.time() - t:.1f}s")

t = time.time()
nx.betweenness_centrality(G, k=400, seed=42)
print(f"betweenness(k=400): {time.time() - t:.1f}s")

t = time.time()
louv = nx.community.louvain_communities(G, seed=42)
print(f"louvain: {len(louv)} communities in {time.time() - t:.1f}s")

print("top degree symbols:", [(anno[n]["symbol"] if n in anno else n, d) for n, d in sorted(G.degree(), key=lambda x: -x[1])[:10]])
print(f"total {time.time() - t0:.1f}s")
