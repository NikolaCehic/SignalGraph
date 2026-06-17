# Retrieval Quality

## Route Distribution

- `comparison`: 15
- `drift`: 85
- `global`: 71
- `hybrid`: 107
- `local`: 70
- `structured_lookup`: 2
- `vector`: 70

## Metrics By Ablation

| Ablation | Context Precision | Context Recall | Required Nodes | Required Edges | Path Validity | Provenance |
|---|---:|---:|---:|---:|---:|---:|
| vector-only | 1.0 | 0.014 | 0.014 | 0.0 | 1.0 | 1.0 |
| hybrid | 1.0 | 0.286 | 0.286 | 0.438 | 1.0 | 1.0 |
| local | 0.996 | 0.321 | 0.321 | 0.667 | 1.0 | 1.0 |
| global | 0.821 | 0.15 | 0.15 | 0.25 | 1.0 | 1.0 |
| DRIFT-style | 0.996 | 0.314 | 0.314 | 0.538 | 1.0 | 1.0 |
| best-route | 1.0 | 0.329 | 0.329 | 0.495 | 1.0 | 1.0 |
