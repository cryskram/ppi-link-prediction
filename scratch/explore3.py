"""Scratch: debug the Entrez ID mapping between the graph and NCBI gene_info."""

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

anno = {}
sym2id = {}
with gzip.open("Homo_sapiens.gene_info.gz", "rt") as f:
    header = f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 11:
            continue
        anno[p[1]] = {"symbol": p[2], "description": p[8]}
        sym2id.setdefault(p[2], p[1])

print("ATM id:", sym2id.get("ATM"), "in graph:", sym2id.get("ATM") in G)
print("TP53 id:", sym2id.get("TP53"), "in graph:", sym2id.get("TP53") in G)
print("BRCA1 id:", sym2id.get("BRCA1"), "in graph:", sym2id.get("BRCA1") in G)

unmatched = [n for n in G if n not in anno]
print("unmatched nodes:", len(unmatched), "sample:", unmatched[:15])
print("matched sample:", [(n, anno[n]["symbol"]) for n in list(G)[:10] if n in anno])

# reverse: do graph ids look like entrez at all?
ints = [n for n in unmatched if n.isdigit()]
print("unmatched numeric:", len(ints), "sample:", ints[:15])
print("all graph ids numeric?", all(n.isdigit() for n in G))

# is ATM (472) somewhere near?
print("472 in graph:", "472" in G)
print("degree of 472 absent -> check 471,473:", "471" in G, "473" in G)
print("number of 1-digit ids in graph:", sum(1 for n in G if len(n) == 1))
print("max node id (numeric):", max((int(n) for n in G if n.isdigit()), default=None))
