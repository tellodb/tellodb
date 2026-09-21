# Baseline

## Canonical protocol

| setting | value |
|---|---|
| split | `dev` |
| client context | `off` |
| timestamps | `session` |
| reset | first run of each suite only |
| seeds | `1..RUNS` |

The results below predate this protocol block and remain stale until E0.3 regenerates them from a clean commit.

## Results

| run | tier | commit | dataset | split | n | err | recall_any (95% CI) | recall_all | nDCG | accuracy (95% CI) | query p50/p95/p99 ms | ingest mem/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1789651919890_longmemeval_s_cleaned_dev_recall | smoke | 878a2b5f* | longmemeval | dev | 10 | 0 | 100.0 (100.0–100.0) | 100.0 | 95.0 | – | 180/285/285 | 62.2 |
| 1789651933190_locomo10_dev_recall | smoke | 878a2b5f* | locomo | dev | 10 | 0 | 100.0 (100.0–100.0) | 100.0 | 74.3 | – | 204/328/328 | 66.2 |
| 1789651945132_synth_1k_all_recall | smoke | 878a2b5f* | longmemeval | all | 60 | 0 | 79.6 (68.5–88.9) | 79.6 | 63.8 | – | 140/298/376 | 190.6 |

### Query Latency Breakdown (Mean ms)

| run | tier | plan | route | embed | ann | fts | cards | rerank | pref | graph | session | fuse | hydr | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1789651919890_longmemeval_s_cleaned_dev_recall | smoke | 3.2 | 4.9 | 0.0 | 2.5 | 9.8 | 12.9 | 0.0 | 0.0 | 138.0 | 11.8 | 0.0 | 0.0 | 203.9 |
| 1789651933190_locomo10_dev_recall | smoke | 6.5 | 8.9 | 0.0 | 0.7 | 3.0 | 9.4 | 0.0 | 0.0 | 163.9 | 4.3 | 0.0 | 0.0 | 216.3 |
| 1789651945132_synth_1k_all_recall | smoke | 0.8 | 1.3 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 | 0.0 | 146.2 | 0.7 | 0.0 | 0.0 | 153.0 |
