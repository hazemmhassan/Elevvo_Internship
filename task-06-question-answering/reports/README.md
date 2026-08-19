# Evaluation reports

`model_comparison.json` was generated with:

```powershell
python scripts\evaluate_models.py --limit 100 --device cpu
```

It contains the SQuAD subset size, timestamp, ranking rule, aggregate Exact
Match/F1/latency for each model, and every question, prediction, reference,
confidence score, per-example metric, and latency measurement.

The model weights and SQuAD dataset remain in the local Hugging Face cache and
are not committed. Re-running the command regenerates the report with fresh
latency measurements.
