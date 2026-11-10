"""Scratch: measure DDR module size as filters tighten."""

import gzip

import networkx as nx

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

anno, sym2id = {}, {}
with gzip.open("Homo_sapiens.gene_info.gz", "rt") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 11:
            continue
        anno[p[1]] = {"symbol": p[2], "description": p[8]}
        sym2id.setdefault(p[2], p[1])

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

seeds = [sym2id[s] for s in DDR if s in sym2id and sym2id[s] in G]
print("seeds in graph:", len(seeds), "of", len(DDR))

seed_set = set(seeds)
expand = set(seed_set)
for s in seeds:
    expand.update(G.neighbors(s))
print("1-hop expansion:", len(expand))

for thresh in (2, 3, 4, 5):
    keep = set(seed_set)
    for v in expand - seed_set:
        if sum(1 for s in seed_set if G.has_edge(v, s)) >= thresh:
            keep.add(v)
    sub = G.subgraph(keep)
    comps = sorted(nx.connected_components(sub), key=len, reverse=True)
    print(f"cohesion>={thresh}: {sub.number_of_nodes()} nodes / {sub.number_of_edges()} edges / largest comp {len(comps[0])}")

# what does degree look like inside the tightest module
keep = set(seed_set)
for v in expand - seed_set:
    if sum(1 for s in seed_set if G.has_edge(v, s)) >= 3:
        keep.add(v)
sub = G.subgraph(keep)
print("module degree max:", max(dict(sub.degree()).values()))
print("module top degree:", [(anno[n]["symbol"] if n in anno else n, d) for n, d in sorted(sub.degree(), key=lambda x: -x[1])[:8]])
pairs = len(keep) * (len(keep) - 1) // 2 - sub.number_of_edges()
print("non-edge pairs in module:", pairs)
