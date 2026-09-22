# pruned (dev, 2026-09-22T08:38Z)

Embed text: context (window 1), client context: off, timestamps: session.

Configurations:
```
baseline
pruned TELLODB_DISABLE=chunks,gist,keywords,session_router,preferences,retrospective_links,derived_links,consolidation,metrics,predicate_canon,graph_edges,facts,relation_companions,semantic_dedup
```

## locomo

Baseline: `/Users/sharjeel/projects/rust/tellodb/benchmarks/runs/20260922_131441_dev_pruned/locomo/baseline/1790065320647_locomo10_dev_recall.json` baseline — recall_any 84.7, nDCG 56.4, ingest 33.0 mem/s, 31271 B/mem, 3.60 embedded/mem, query p95 1992 ms

Deltas are paired over questions against `baseline`. p-values are two-sided percentile bootstrap, Holm-Bonferroni corrected across the 1 arms separately for each metric; "keep" means an adjusted p below 0.05 on either metric.

| config | n paired | Δ recall_any (95% CI) | p adj | Δ nDCG (95% CI) | p adj | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pruned | 432 | -6.5 (-7.5…-4.6) | <0.001 | -5.6 (-6.4…-4.2) | <0.001 | +4072.1% | -48.3% | -0.3% | -45.4% | 60% | keep |
