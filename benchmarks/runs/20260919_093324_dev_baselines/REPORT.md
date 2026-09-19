# baselines (dev, 2026-09-19T11:59Z)

Embed text: context (window 1), client context: window, timestamps: session.

Configurations:
```
full
bm25_only TELLODB_LANES=fts
vector_only TELLODB_LANES=vector
hybrid_rrf TELLODB_LANES=vector,fts
hybrid_rerank TELLODB_LANES=vector,fts,rerank
no_graph TELLODB_LANES=vector,fts,cards,rerank,route
no_rerank TELLODB_LANES=vector,fts,cards,graph,route
```
Deltas are paired over questions against `full`; "drop" means neither quality delta's 95% CI excludes zero.

## longmemeval

Baseline: `/root/tellodb-instr/benchmarks/runs/20260919_093324_dev_baselines/longmemeval/full/1789811487535_longmemeval_s_cleaned_dev_recall.json` full — recall_any 91.9, nDCG 81.3, ingest 75.1 mem/s, 91853 B/mem, 5.48 embedded/mem, query p95 856 ms

| config | n paired | Δ recall_any (95% CI) | Δ nDCG (95% CI) | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |
|---|---|---|---|---|---|---|---|---|---|
| bm25_only | 149 | +1.3 (-2.0…+4.7) | +1.1 (-1.3…+3.5) | -6.3% | -0.0% | +0.0% | -53.0% | 0% | drop |
| vector_only | 149 | -3.4 (-6.7…+0.0) | -6.1 (-9.1…-3.6) | -15.4% | +0.0% | +0.1% | -76.3% | 0% | keep |
| hybrid_rrf | 149 | -2.7 (-6.0…+0.0) | -4.4 (-7.2…-2.1) | -13.1% | +0.0% | +0.1% | -49.5% | 0% | keep |
| hybrid_rerank | 149 | -2.7 (-6.0…+0.0) | -2.3 (-4.7…-0.3) | -15.4% | +0.0% | +0.1% | -26.8% | 93% | keep |
| no_graph | 149 | -2.7 (-5.4…-0.7) | -2.0 (-4.2…-0.2) | -16.1% | +0.0% | +0.1% | -5.7% | 93% | keep |
| no_rerank | 149 | -0.7 (-2.0…+0.0) | -1.2 (-2.4…-0.1) | -12.4% | +0.0% | +0.1% | -21.6% | 0% | keep |

## locomo

Baseline: `/root/tellodb-instr/benchmarks/runs/20260919_093324_dev_baselines/locomo/full/1789811613461_locomo10_dev_recall.json` full — recall_any 93.1, nDCG 71.4, ingest 62.2 mem/s, 4950 B/mem, 3.99 embedded/mem, query p95 365 ms

| config | n paired | Δ recall_any (95% CI) | Δ nDCG (95% CI) | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |
|---|---|---|---|---|---|---|---|---|---|
| bm25_only | 432 | -3.7 (-6.0…-1.6) | -3.1 (-5.7…-0.6) | +3.1% | -100.0% | +0.2% | -62.2% | 0% | keep |
| vector_only | 432 | -8.6 (-12.3…-5.3) | -10.5 (-13.5…-7.6) | +4.4% | +2.8% | +0.2% | -86.3% | 0% | keep |
| hybrid_rrf | 432 | -4.4 (-7.6…-1.4) | -5.6 (-8.3…-2.9) | +3.5% | -23.1% | +0.1% | -59.2% | 0% | keep |
| hybrid_rerank | 432 | -3.2 (-6.5…+0.0) | -3.9 (-6.4…-1.3) | +4.8% | -100.0% | +0.1% | -30.4% | 89% | keep |
| no_graph | 432 | -1.4 (-2.8…-0.2) | -3.6 (-5.4…-1.9) | +3.2% | -40.2% | +0.1% | -2.5% | 89% | keep |
| no_rerank | 432 | -0.9 (-2.1…+0.2) | -1.7 (-3.0…-0.4) | +3.1% | -5.1% | +0.2% | -27.4% | 0% | keep |
