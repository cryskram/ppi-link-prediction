# Protein–Protein Interaction Link Prediction — SNA Alternate Assessment

**Theme:** Protein–Protein Interaction Prediction Network
**Deliverable 1 (code):** `PPI_Link_Prediction_SNA.ipynb` — fully executed, 24 code cells, 9 embedded figures.
**Runtime:** ~4 minutes end-to-end from a cold start on a laptop CPU.

---

## What the project does

The human interactome is an incomplete map. This project treats it as a social network of proteins
and asks which pairs are likely to interact but have not been recorded yet — then ranks those pairs so
a lab can decide what to test next.

1. **Problem & modelling** — proteins as nodes, documented physical interactions as undirected edges.
2. **Data & construction** — BioSNAP human PPI network (21,521 proteins / 338,625 interactions) joined
   with NCBI Gene annotations; 80/20 edge split; three families of negative samples.
3. **SNA analysis** — degree distribution (γ), clustering, assortativity, path length, k-core,
   centrality, communities.
4. **Link prediction** — Common Neighbours, Jaccard, Adamic–Adar, Preferential Attachment, all
   implemented from scratch and cross-checked against NetworkX.
5. **Comparison & validation** — Precision@K, Recall@K, F1@K, ROC-AUC, PR-AUC across three negative
   regimes, plus a 3-split stability check.
6. **Recommendation & interpretation** — hub-capped top-10 predictions for the DNA damage response
   (DDR) core, with the shared-partner evidence behind each one, exported to Cytoscape.

## Headline results

| Regime | ROC-AUC winner | Precision/PR-AUC winner |
|---|---|---|
| R1 random negatives | Preferential Attachment (0.920) | Preferential Attachment (0.762) |
| R2 degree-matched negatives (fair) | Jaccard (0.730) | Jaccard (0.765), Adamic–Adar 0.750 |
| R3 2-hop hard negatives | Preferential Attachment (0.553, marginal) | **Common Neighbours** (P@1,000 = 0.83 vs PA's 0.177) |

**Conclusion:** Preferential Attachment's apparent strength is a degree artefact — it collapses from
AUC 0.92 to 0.55 once negatives are degree-matched. Adamic–Adar is the recommended default; it
recovers held-out DDR interactions ≈ 26× better than chance inside the DDR core.

## Files

```
PPI_Link_Prediction_SNA.ipynb     executed notebook (Deliverable 1)
data/
  PP-Pathways_ppi.csv.gz          raw source: BioSNAP human PPI edge list
  Homo_sapiens.gene_info.gz       raw source: NCBI Gene annotations
  ppi_edges_clean.csv             cleaned edge list with gene symbols
  node_attributes.csv             node table: symbol, name, degree, clustering, k-core, community
figures/
  01_core_interactome.png         core interactome, coloured by Louvain community
  02_degree_distribution.png      degree CCDF + power-law estimates
  03_kcore_profile.png            k-core decomposition
  04_roc_pr_curves.png            ROC + PR curves for all four indices, all three regimes
  05_precision_at_k.png           Precision@K / F1@K curves + heatmap
  06_evidence_distributions.png   common-neighbour evidence, positives vs negatives
  07_stability.png                AUC across three independent splits
  08_ddr_enrichment.png           DDR-focused enrichment over chance
  09_ddr_module_predictions.png   DDR core + predictions, and the evidence behind each prediction
  ddr_module.cyjs                 Cytoscape native JSON (nodes + edges + attributes)
  ddr_module.graphml              GraphML for Cytoscape / Gephi
  ddr_module.sif                  SIF edge list for quick Cytoscape import
  ddr_module_interactive.html     drag-and-drop interactive network (open in a browser)
outputs/
  metrics_all_regimes.csv         every metric, every method, every regime
  metrics_stability.csv           mean ± std over three splits
  predicted_links_top20.csv       raw top-20 Adamic–Adar predictions
  predicted_links_top10_hub_capped.csv  the shortlist for wet-lab follow-up
  ddr_validation.csv              held-out recovery at K = 100 … 5,000
  ddr_candidate_scores.csv        all DDR candidate pairs with their four scores
  ddr_module_nodes.csv            module node attributes (for Cytoscape table import)
```

## How to run

```bash
pip install networkx pandas numpy scikit-learn matplotlib seaborn pyvis
jupyter notebook PPI_Link_Prediction_SNA.ipynb     # Run All
```

The two source files are downloaded automatically if `data/` is empty, so the notebook also runs
top-to-bottom in Google Colab without any manual upload.

## Cytoscape

Open `figures/ddr_module.cyjs` (or `.graphml`) with **File ▸ Import ▸ Network from File…**, then
import `outputs/ddr_module_nodes.csv` with **File ▸ Import ▸ Table from File…** to attach attributes.

Suggested style mappings: node size → `degree`, fill colour → `community`, border width → `is_ddr_seed`,
label → `symbol`, edge line style → `predicted` (dashed for predictions), edge width → `weight`.

## Data sources

- Agrawal M., Zitnik M., Leskovec J. *Large-scale analysis of disease pathways in the human
  interactome.* Pacific Symposium on Biocomputing, 2018. Dataset: BioSNAP (Stanford SNAP).
- NCBI Gene: `Homo_sapiens.gene_info.gz` (Entrez ID → symbol, description, chromosome, gene type).

## Limitations

Unweighted binary edges (no confidence scores), edge-based rather than temporal split, negatives are
"not yet recorded" rather than proven non-interactions, a curated (not KEGG-pulled) DDR seed list, a
hard hub cap as a blunt artefact filter, and no wet-lab validation — every recommendation is a ranked
hypothesis.
