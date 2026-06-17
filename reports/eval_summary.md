# SignalGraph Eval Summary

- Questions: 70
- Ablations: vector-only, hybrid, local, global, DRIFT-style, best-route
- Result rows: 420

## Category Counts

- `entity-specific`: 15
- `comparison`: 15
- `broad landscape`: 15
- `structured`: 10
- `decision-memo`: 10
- `adversarial/uncertainty`: 5

## Graph Metrics

- `required_node_recall`: 0.236
- `required_edge_recall`: 0.398
- `path_validity`: 1.0
- `provenance_coverage`: 1.0
- `merge_quality`: 1.0
- `staleness_detection`: 0.976

## Ablation Summary

| Ablation | Faithfulness | Context Recall | Required Node Recall | Required Edge Recall | Path Validity | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vector-only | 1.0 | 0.014 | 0.014 | 0.0 | 1.0 | 2.35 |
| hybrid | 1.0 | 0.286 | 0.286 | 0.438 | 1.0 | 61.515 |
| local | 1.0 | 0.321 | 0.321 | 0.667 | 1.0 | 163.503 |
| global | 1.0 | 0.15 | 0.15 | 0.25 | 1.0 | 2.801 |
| DRIFT-style | 1.0 | 0.314 | 0.314 | 0.538 | 1.0 | 299.812 |
| best-route | 1.0 | 0.329 | 0.329 | 0.495 | 1.0 | 123.568 |

## Lowest Scoring Cases

| ID | Category | Ablation | Required Nodes | Required Edges | Path Validity |
|---|---|---|---:|---:|---:|
| ENT-01 | entity-specific | vector-only | 0.0 | 0.0 | 1.0 |
| ENT-01 | entity-specific | global | 0.0 | 0.0 | 1.0 |
| ENT-02 | entity-specific | vector-only | 0.0 | 0.0 | 1.0 |
| ENT-02 | entity-specific | global | 0.0 | 0.0 | 1.0 |
| ENT-03 | entity-specific | vector-only | 0.0 | 0.0 | 1.0 |
| ENT-03 | entity-specific | global | 0.0 | 0.0 | 1.0 |
| ENT-04 | entity-specific | vector-only | 0.0 | 0.0 | 1.0 |
| ENT-04 | entity-specific | global | 0.0 | 0.0 | 1.0 |
| ENT-05 | entity-specific | vector-only | 0.0 | 0.0 | 1.0 |
| ENT-05 | entity-specific | global | 0.0 | 0.0 | 1.0 |
