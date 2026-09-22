# groups (dev, 2026-09-22T19:09Z)

Embed text: context (window 1), client context: off, timestamps: session.

Configurations:
```
baseline
g_companions TELLODB_DISABLE=gist,keywords,relation_companions
g_links TELLODB_DISABLE=retrospective_links,derived_links,graph_edges
g_routing TELLODB_DISABLE=session_router,preferences
g_facts TELLODB_DISABLE=facts,predicate_canon,semantic_dedup,consolidation
g_misc TELLODB_DISABLE=chunks,metrics
pruned_all TELLODB_DISABLE=chunks,gist,keywords,session_router,preferences,retrospective_links,derived_links,consolidation,metrics,predicate_canon,graph_edges,facts,relation_companions,semantic_dedup
```

## locomo

Baseline: `/root/tellodb/benchmarks/runs/20260922_185547_dev_groups/locomo/baseline/1790103416478_locomo10_dev_recall.json` baseline — recall_any 86.1, nDCG 57.2, ingest 68.7 mem/s, 31161 B/mem, 3.70 embedded/mem, query p95 166 ms

Deltas are paired over questions against `baseline`, with each arm's per-question scores averaged over its 3 seed(s) first. p-values are two-sided percentile bootstrap, Holm-Bonferroni corrected across the 6 arms separately for each metric; "keep" means an adjusted p below 0.05 on either metric.

| config | n paired | Δ recall_any (95% CI) | p adj | Δ nDCG (95% CI) | p adj | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g_companions | 432 | -0.8 (-1.3…+1.2) | 1.000 | -0.7 (-1.5…+0.8) | 1.000 | +31.8% | -12.6% | -1.3% | -21.7% | 60% | drop |
| g_links | 432 | -6.6 (-8.6…-3.1) | <0.001 | -6.3 (-8.2…-5.4) | <0.001 | -10.5% | -5.7% | +0.0% | -31.3% | 60% | keep |
| g_routing | 432 | -0.9 (-4.5…+3.7) | 1.000 | +1.6 (-2.5…+7.1) | 1.000 | -53.0% | -30.1% | +0.0% | -22.9% | 60% | drop |
| g_facts | 432 | +0.0 (-4.9…+1.5) | 1.000 | +1.2 (-2.3…+3.0) | 1.000 | -14.4% | -5.6% | +0.0% | -19.9% | 60% | drop |
| g_misc | 432 | +0.0 (+0.0…+0.0) | 1.000 | +0.0 (+0.0…+0.0) | 1.000 | -11.3% | +0.0% | +0.0% | -19.3% | 60% | drop |
| pruned_all | 432 | -7.3 (-8.9…-4.4) | <0.001 | -6.7 (-8.1…-4.7) | <0.001 | +5.9% | -47.7% | -1.3% | -32.5% | 60% | keep |
