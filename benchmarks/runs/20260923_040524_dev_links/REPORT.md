# links (dev, 2026-09-23T04:21Z)

Embed text: context (window 1), client context: off, timestamps: session.

Configurations:
```
baseline
only_graph_edges TELLODB_DISABLE=retrospective_links,derived_links
only_retrospective TELLODB_DISABLE=derived_links,graph_edges
only_derived TELLODB_DISABLE=retrospective_links,graph_edges
no_retrospective TELLODB_DISABLE=retrospective_links
no_derived TELLODB_DISABLE=derived_links
no_graph_edges TELLODB_DISABLE=graph_edges
g_links TELLODB_DISABLE=retrospective_links,derived_links,graph_edges
```

## locomo

Baseline: `/root/tellodb/benchmarks/runs/20260923_040524_dev_links/locomo/baseline/1790136389311_locomo10_dev_recall.json` baseline — recall_any 86.1, nDCG 57.2, ingest 59.2 mem/s, 31161 B/mem, 3.70 embedded/mem, query p95 117 ms

Deltas are paired over questions against `baseline`, with each arm's per-question scores averaged over its 3 seed(s) first. p-values are two-sided percentile bootstrap, Holm-Bonferroni corrected across the 7 arms separately for each metric; "keep" means an adjusted p below 0.05 on either metric.

| config | n paired | Δ recall_any (95% CI) | p adj | Δ nDCG (95% CI) | p adj | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| only_graph_edges | 432 | -0.9 (-2.0…+0.0) | 1.000 | -0.9 (-1.6…-0.1) | <0.001 | +0.1% | -5.7% | +0.0% | +9.4% | 60% | keep |
| only_retrospective | 432 | -6.6 (-8.6…-3.1) | <0.001 | -6.3 (-8.2…-5.4) | <0.001 | -16.0% | -5.7% | +0.0% | -17.1% | 60% | keep |
| only_derived | 432 | -2.9 (-6.2…+1.5) | 0.540 | +1.0 (+0.5…+1.8) | <0.001 | +1.6% | +0.0% | +0.0% | -12.8% | 60% | keep |
| no_retrospective | 432 | +0.0 (+0.0…+0.0) | 1.000 | +0.0 (+0.0…+0.0) | 1.000 | +5.2% | +0.0% | +0.0% | +14.5% | 60% | drop |
| no_derived | 432 | -0.9 (-2.0…+0.0) | 1.000 | -0.9 (-1.6…-0.1) | <0.001 | +1.8% | -5.7% | +0.0% | -0.9% | 60% | keep |
| no_graph_edges | 432 | -2.9 (-6.2…+1.5) | 0.540 | +1.0 (+0.5…+1.8) | <0.001 | -8.5% | +0.0% | +0.0% | -6.0% | 60% | keep |
| g_links | 432 | -6.6 (-8.6…-3.1) | <0.001 | -6.3 (-8.2…-5.4) | <0.001 | -18.9% | -5.7% | +0.0% | -2.6% | 60% | keep |
