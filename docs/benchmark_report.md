# TraceMind Benchmark Report

Generated: `2026-07-26T23:10:15.832588+00:00`<br>
Dataset: `data/benchmark.jsonl` (18 tasks)<br>
Judge: `expected-output-regex`<br>
Reflector: `dataset`

## Aggregate results

| Mode | Success | Success rate | Avg. healing attempts | Avg. raw tokens | Avg. retained tokens | Token reduction | Avg. duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-pass | 6/18 | 33.33% | 0.00 | 196.32 | 196.32 | 0.00% | 0.228s |
| TraceMind self-healing | 18/18 | 100.00% | 0.67 | 962.33 | 830.47 | 13.70% | 0.378s |

**Measured completion-rate lift: +66.67 percentage points.**

## Task-level results

| Task | Category | Single-pass | Self-healing | Repairs |
|---|---|---:|---:|---:|
| `missing-dict-column` | missing column | ❌ | ✅ | 1 |
| `division-by-zero` | arithmetic edge case | ❌ | ✅ | 1 |
| `invalid-api-schema` | schema mismatch | ❌ | ✅ | 1 |
| `index-off-by-one` | index bounds | ❌ | ✅ | 1 |
| `missing-third-party-package` | dependency failure | ❌ | ✅ | 1 |
| `syntax-missing-colon` | syntax error | ❌ | ✅ | 1 |
| `json-key-mismatch` | schema mismatch | ❌ | ✅ | 1 |
| `none-arithmetic` | null handling | ❌ | ✅ | 1 |
| `date-format-mismatch` | data parsing | ❌ | ✅ | 1 |
| `csv-column-mismatch` | missing column | ❌ | ✅ | 1 |
| `empty-maximum` | empty collection | ❌ | ✅ | 1 |
| `numeric-string-sum` | type mismatch | ❌ | ✅ | 1 |
| `fibonacci-baseline` | baseline algorithm | ✅ | ✅ | 0 |
| `unique-sort-baseline` | baseline transformation | ✅ | ✅ | 0 |
| `word-count-baseline` | baseline text | ✅ | ✅ | 0 |
| `median-baseline` | baseline statistics | ✅ | ✅ | 0 |
| `group-aggregate-baseline` | baseline aggregation | ✅ | ✅ | 0 |
| `svg-artifact-baseline` | baseline artifact | ✅ | ✅ | 0 |

## Methodology

- Both modes execute inside the same offline Docker sandbox.
- Single-pass executes the generated program once with no correction.
- Self-healing uses the full LangGraph retry path with deterministic dataset-backed exact-text patches for reproducibility.
- Context tokens are estimated at four characters per token; raw and pruned maxima are measured immediately around each pruning step.
- The committed report uses exact expected-output oracles. Use --judge ollama for optional local semantic LLM judging.
- This fixture benchmark is evidence for these tasks, not a universal model claim.

Reproduce:

```bash
python -m src.eval.benchmark
```
