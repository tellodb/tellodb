# heuristics (dev, 2026-09-19T12:46Z)

Embed text: context (window 1), client context: window, timestamps: session.

Configurations:
```
generic TELLODB_HEURISTICS=generic
legacy_tuned TELLODB_HEURISTICS=legacy-tuned
```
Deltas are paired over questions against `generic`; "drop" means neither quality delta's 95% CI excludes zero.

## longmemeval

Baseline: `/root/tellodb-instr/benchmarks/runs/20260919_120500_dev_heuristics/longmemeval/generic/1789820625400_longmemeval_s_cleaned_dev_recall.json` generic — recall_any 91.9, nDCG 81.4, ingest 72.0 mem/s, 91865 B/mem, 5.48 embedded/mem, query p95 861 ms

| config | n paired | Δ recall_any (95% CI) | Δ nDCG (95% CI) | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |
|---|---|---|---|---|---|---|---|---|---|
| legacy_tuned | 149 | +0.0 (+0.0…+0.0) | +0.0 (+0.0…+0.0) | -1.4% | -0.0% | -0.0% | +2.1% | 93% | drop |

## locomo

Baseline: `/root/tellodb-instr/benchmarks/runs/20260919_120500_dev_heuristics/locomo/generic/1789820752397_locomo10_dev_recall.json` generic — recall_any 93.1, nDCG 71.3, ingest 62.9 mem/s, 0 B/mem, 3.99 embedded/mem, query p95 373 ms

| config | n paired | Δ recall_any (95% CI) | Δ nDCG (95% CI) | Δ ingest mem/s | Δ bytes/mem | Δ embedded/mem | Δ query p95 | rerank rate | keep? |
|---|---|---|---|---|---|---|---|---|---|
| legacy_tuned | 432 | -0.2 (-0.7…+0.0) | -0.1 (-0.3…+0.1) | +2.2% | – | +0.0% | -4.8% | 89% | drop |
