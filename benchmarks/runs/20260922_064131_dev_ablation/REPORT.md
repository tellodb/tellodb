# ablation (dev, 2026-09-22T07:03Z)

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
Deltas are paired over questions against `baseline`; "drop" means neither quality delta's 95% CI excludes zero.

## locomo

Baseline: `/root/tellodb/benchmarks/runs/20260922_064131_dev_ablation/locomo/baseline/1790059365628_locomo10_dev_recall.json` baseline — recall_any 85.0, nDCG 57.0, ingest 88.3 mem/s, 31221 B/mem, 3.62 embedded/mem, query p95 69 ms

Deltas are paired over questions against `baseline`. p-values are two-sided percentile bootstrap, Holm-Bonferroni corrected across the 18 arms separately for each metric; "keep" means an adjusted p below 0.05 on either metric.

| config | n paired | Δ recall_any (95% CI) | p adj | Δ nDCG (95% CI) | p adj | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no_chunks | 432 | +0.9 (+0.5…+1.3) | <0.001 | +0.3 (-0.1…+1.5) | 1.000 | +51.6% | +0.2% | +1.8% | -7.2% | 59% | keep |
| no_gist | 432 | +0.7 (+0.0…+1.3) | 0.480 | -0.3 (-0.8…+1.5) | 1.000 | +45.7% | -0.9% | +0.5% | -8.7% | 59% | drop |
| no_keywords | 432 | +0.7 (-0.5…+2.0) | 1.000 | -0.9 (-1.6…+0.1) | 0.984 | +63.6% | -0.7% | +1.5% | -8.7% | 59% | drop |
| no_fact_companions | 432 | +0.2 (-3.7…+1.3) | 1.000 | -1.2 (-3.2…+0.5) | 0.780 | +76.8% | -19.9% | -16.2% | -13.0% | 58% | drop |
| no_atomic_cards | 432 | -1.2 (-1.3…-1.0) | <0.001 | -2.4 (-3.7…-1.7) | <0.001 | +79.5% | -14.8% | -22.6% | -26.1% | 59% | keep |
| no_event_companions | 432 | -4.4 (-7.4…-3.0) | <0.001 | -5.1 (-5.9…-4.4) | <0.001 | +77.1% | -27.8% | -26.0% | -8.7% | 57% | keep |
| no_relation_companions | 432 | +2.1 (+1.5…+3.7) | <0.001 | +0.9 (-0.8…+2.2) | 1.000 | +37.7% | -10.6% | +1.6% | -10.1% | 59% | keep |
| no_memory_cards | 432 | -1.2 (-1.5…-0.7) | <0.001 | +1.9 (+0.5…+3.4) | <0.001 | +79.8% | -31.1% | +2.0% | -29.0% | 59% | keep |
| no_session_router | 432 | +0.7 (-3.0…+6.2) | 1.000 | +1.9 (-1.3…+8.9) | 1.000 | +116.1% | -29.7% | +1.5% | -17.4% | 59% | drop |
| no_preferences | 432 | +0.5 (+0.0…+0.7) | 0.574 | -0.0 (-0.7…+0.5) | 1.000 | +52.8% | -0.0% | +1.6% | -7.2% | 59% | drop |
| no_retrospective_links | 432 | +0.9 (+0.7…+1.2) | <0.001 | +0.2 (-0.6…+1.0) | 1.000 | +65.5% | +0.1% | +1.7% | -7.2% | 59% | keep |
| no_derived_links | 432 | +0.0 (-1.0…+1.2) | 1.000 | -0.5 (-1.0…+1.0) | 1.000 | +37.6% | -5.5% | +1.7% | -7.2% | 59% | drop |
| no_graph_edges | 432 | -1.9 (-4.9…+2.0) | 1.000 | +1.3 (+0.8…+2.7) | <0.001 | +39.8% | +0.1% | +1.7% | -30.4% | 59% | keep |
| no_facts | 432 | +1.6 (-2.5…+3.0) | 1.000 | +1.6 (-0.6…+3.4) | 1.000 | +83.6% | -5.7% | +1.6% | -11.6% | 60% | drop |
| no_semantic_dedup | 432 | +2.1 (+1.2…+2.5) | <0.001 | +0.8 (+0.1…+1.4) | <0.001 | +68.8% | +0.1% | +1.8% | -8.7% | 60% | keep |
| no_consolidation | 432 | +1.4 (+1.2…+1.5) | <0.001 | +0.5 (-0.6…+1.4) | 1.000 | +36.1% | -0.0% | +1.8% | -5.8% | 60% | keep |
| no_metrics | 432 | +0.9 (+0.7…+1.2) | <0.001 | +0.3 (-0.4…+1.5) | 1.000 | +36.6% | -0.2% | +1.8% | -7.2% | 60% | keep |
| no_predicate_canon | 432 | +1.6 (+1.2…+2.0) | <0.001 | +0.5 (-0.4…+1.9) | 1.000 | +74.5% | -1.8% | +1.9% | -11.6% | 59% | keep |
