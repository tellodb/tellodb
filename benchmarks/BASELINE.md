# Baseline

## Canonical protocol

| setting | value |
|---|---|
| split | `dev` |
| client context | `off` |
| timestamps | `session` |
| reset | first run of each suite only |
| seeds | `1..RUNS` |

## Results

| run | seed | tier | commit | dataset | split | n (bootstrap unit) | err | recall_any (95% CI) | recall_all | nDCG | accuracy (95% CI) | query p50/p95/p99 ms | ingest mem/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1790005171111_longmemeval_s_cleaned_dev_recall | 1 | dev | d508051f | longmemeval | dev | 149 questions | 0 | 94.0 (89.9–97.3) | 81.9 | 82.3 | – | 144/303/370 | 28.7 |
| 1790005282316_longmemeval_s_cleaned_dev_recall | 2 | dev | d508051f | longmemeval | dev | 149 questions | 0 | 95.3 (91.9–98.0) | 84.6 | 83.8 | – | 173/362/435 | 897.7 |
| 1790005384948_longmemeval_s_cleaned_dev_recall | 3 | dev | d508051f | longmemeval | dev | 149 questions | 0 | 95.3 (91.3–98.7) | 84.6 | 83.8 | – | 156/350/421 | 979.1 |
| 1790005432865_locomo10_dev_recall | 1 | dev | d508051f | locomo | dev | 3 conversations | 0 | 85.9 (82.9–88.8) | 72.7 | 56.6 | – | 65/142/171 | 155.8 |
| 1790005461988_locomo10_dev_recall | 2 | dev | d508051f | locomo | dev | 3 conversations | 0 | 86.1 (83.4–88.8) | 74.3 | 57.3 | – | 59/133/163 | 1682.9 |
| 1790005486261_locomo10_dev_recall | 3 | dev | d508051f | locomo | dev | 3 conversations | 0 | 86.1 (83.4–88.8) | 74.3 | 57.3 | – | 46/108/136 | 1244.1 |

### Configuration

| run | client context | timestamps | heuristics | lanes | rerank | embed cache hit % | device |
|---|---|---|---|---|---|---|---|
| 1790005171111_longmemeval_s_cleaned_dev_recall | off | session | generic | vector+fts+cards+rerank+graph+route | BAAI/bge-reranker-base | 12.7 | CUDA |
| 1790005282316_longmemeval_s_cleaned_dev_recall | off | session | generic | vector+fts+cards+rerank+graph+route | BAAI/bge-reranker-base | 13.8 | CUDA |
| 1790005384948_longmemeval_s_cleaned_dev_recall | off | session | generic | vector+fts+cards+rerank+graph+route | BAAI/bge-reranker-base | 17.6 | CUDA |
| 1790005432865_locomo10_dev_recall | off | session | generic | vector+fts+cards+rerank+graph+route | BAAI/bge-reranker-base | 17.4 | CUDA |
| 1790005461988_locomo10_dev_recall | off | session | generic | vector+fts+cards+rerank+graph+route | BAAI/bge-reranker-base | 17.6 | CUDA |
| 1790005486261_locomo10_dev_recall | off | session | generic | vector+fts+cards+rerank+graph+route | BAAI/bge-reranker-base | 17.8 | CUDA |

### Query Latency Breakdown (Mean ms)

| run | tier | plan | route | embed | ann | fts | cards | rerank | pref | graph | session | fuse | hydr | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1790005171111_longmemeval_s_cleaned_dev_recall | dev | 30.6 | 13.8 | 2.1 | 1.3 | 20.5 | 0.0 | 31.8 | 0.0 | 19.8 | 21.9 | 0.0 | 2.8 | 148.5 |
| 1790005282316_longmemeval_s_cleaned_dev_recall | dev | 50.2 | 21.5 | 0.0 | 1.1 | 33.0 | 0.0 | 11.3 | 0.0 | 23.7 | 40.0 | 0.0 | 2.6 | 188.0 |
| 1790005384948_longmemeval_s_cleaned_dev_recall | dev | 51.0 | 21.6 | 0.0 | 1.0 | 33.0 | 0.0 | 0.1 | 0.0 | 23.6 | 39.1 | 0.0 | 2.6 | 176.6 |
| 1790005432865_locomo10_dev_recall | dev | 5.7 | 10.3 | 2.1 | 0.5 | 19.2 | 0.0 | 12.4 | 0.0 | 11.1 | 3.7 | 0.0 | 1.0 | 69.3 |
| 1790005461988_locomo10_dev_recall | dev | 3.2 | 10.1 | 0.0 | 0.5 | 19.0 | 0.0 | 10.2 | 0.0 | 11.0 | 3.7 | 0.0 | 1.0 | 62.9 |
| 1790005486261_locomo10_dev_recall | dev | 3.1 | 9.4 | 0.0 | 0.4 | 17.6 | 0.0 | 0.0 | 0.0 | 11.2 | 3.8 | 0.0 | 1.1 | 50.6 |
