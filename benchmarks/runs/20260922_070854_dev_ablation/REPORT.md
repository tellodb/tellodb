# ablation (dev, 2026-09-22T07:35Z)

Embed text: context (window 1), client context: off, timestamps: session.

Configurations:
```
baseline
no_chunks TELLODB_DISABLE=chunks
no_gist TELLODB_DISABLE=gist
no_keywords TELLODB_DISABLE=keywords
no_fact_companions TELLODB_DISABLE=fact_companions
no_atomic_cards TELLODB_DISABLE=atomic_cards
no_event_companions TELLODB_DISABLE=event_companions
no_relation_companions TELLODB_DISABLE=relation_companions
no_memory_cards TELLODB_DISABLE=memory_cards
no_session_router TELLODB_DISABLE=session_router
no_preferences TELLODB_DISABLE=preferences
no_retrospective_links TELLODB_DISABLE=retrospective_links
no_derived_links TELLODB_DISABLE=derived_links
no_graph_edges TELLODB_DISABLE=graph_edges
no_facts TELLODB_DISABLE=facts
no_semantic_dedup TELLODB_DISABLE=semantic_dedup
no_consolidation TELLODB_DISABLE=consolidation
no_metrics TELLODB_DISABLE=metrics
no_predicate_canon TELLODB_DISABLE=predicate_canon
```

## locomo

Baseline: `/root/tellodb/benchmarks/runs/20260922_070854_dev_ablation/locomo/baseline/1790060972650_locomo10_dev_recall.json` baseline — recall_any 86.1, nDCG 57.1, ingest 117.3 mem/s, 31216 B/mem, 3.61 embedded/mem, query p95 68 ms

Deltas are paired over questions against `baseline`. p-values are two-sided percentile bootstrap, Holm-Bonferroni corrected across the 18 arms separately for each metric; "keep" means an adjusted p below 0.05 on either metric.

| config | n paired | Δ recall_any (95% CI) | p adj | Δ nDCG (95% CI) | p adj | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no_chunks | 432 | +0.5 (+0.0…+1.2) | 0.900 | +0.5 (-0.0…+2.2) | 1.000 | +11.1% | +0.3% | +1.9% | -4.4% | 59% | drop |
| no_gist | 432 | -0.5 (-2.0…+1.3) | 1.000 | -0.9 (-1.6…+1.6) | 1.000 | +13.7% | -1.2% | +0.6% | -8.8% | 59% | drop |
| no_keywords | 432 | -0.5 (-2.5…+1.3) | 1.000 | -0.9 (-2.4…+2.2) | 1.000 | +30.5% | -0.3% | +2.0% | -5.9% | 59% | drop |
| no_fact_companions | 432 | -0.9 (-3.7…+1.3) | 1.000 | -0.9 (-2.8…-0.3) | <0.001 | +54.5% | -20.0% | -15.8% | -11.8% | 59% | keep |
| no_atomic_cards | 432 | -1.9 (-2.5…-1.2) | <0.001 | -2.3 (-3.4…-0.7) | <0.001 | +25.7% | -14.5% | -22.7% | -22.1% | 59% | keep |
| no_event_companions | 432 | -5.1 (-5.3…-4.9) | <0.001 | -5.2 (-6.0…-3.1) | <0.001 | +37.1% | -28.2% | -25.7% | -10.3% | 58% | keep |
| no_relation_companions | 432 | +0.9 (+0.0…+2.5) | 0.900 | +1.1 (+0.1…+2.5) | <0.001 | +37.6% | -10.7% | +1.8% | -8.8% | 60% | keep |
| no_memory_cards | 432 | -2.3 (-4.0…-0.7) | <0.001 | +1.7 (+0.7…+3.8) | <0.001 | +33.4% | -30.8% | +2.1% | -30.9% | 59% | keep |
| no_session_router | 432 | -1.2 (-6.0…+4.9) | 1.000 | +2.1 (-1.8…+9.5) | 1.000 | +77.4% | -29.7% | +1.9% | -16.2% | 59% | drop |
| no_preferences | 432 | +0.2 (-1.0…+2.5) | 1.000 | +0.3 (-0.6…+2.7) | 1.000 | +16.1% | -0.1% | +2.0% | -5.9% | 60% | drop |
| no_retrospective_links | 432 | +0.2 (-1.0…+1.3) | 1.000 | +0.3 (-0.4…+2.5) | 1.000 | +25.6% | -0.1% | +2.0% | -5.9% | 59% | drop |
| no_derived_links | 432 | -0.9 (-3.0…+1.2) | 1.000 | -0.7 (-1.6…+1.7) | 1.000 | +15.7% | -6.1% | +1.9% | -7.4% | 59% | drop |
| no_graph_edges | 432 | -3.0 (-6.0…+2.0) | 1.000 | +1.3 (+0.6…+3.1) | <0.001 | +8.8% | +0.3% | +2.0% | -30.9% | 58% | keep |
| no_facts | 432 | +0.0 (-3.7…+1.3) | 1.000 | +1.6 (+0.3…+3.2) | <0.001 | +15.2% | -5.7% | +2.1% | -7.4% | 58% | keep |
| no_semantic_dedup | 432 | +0.9 (-0.5…+2.5) | 1.000 | +0.7 (+0.2…+2.7) | <0.001 | +13.9% | +0.1% | +1.7% | -4.4% | 59% | keep |
| no_consolidation | 432 | -0.2 (-1.5…+1.2) | 1.000 | +0.2 (-0.2…+1.4) | 1.000 | +10.4% | +0.4% | +2.0% | -10.3% | 59% | drop |
| no_metrics | 432 | -0.5 (-2.5…+1.3) | 1.000 | -0.0 (-0.8…+2.4) | 1.000 | +18.8% | +0.1% | +2.0% | -8.8% | 59% | drop |
| no_predicate_canon | 432 | +0.0 (-2.0…+2.0) | 1.000 | +0.2 (-0.6…+2.3) | 1.000 | +30.1% | -1.2% | +2.0% | -4.4% | 59% | drop |
