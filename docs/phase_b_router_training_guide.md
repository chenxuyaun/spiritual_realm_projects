# Phase B Router Training Guide

## Overview

Phase B includes three router versions that evolve from rule-based to ML-driven routing:
- **Router v1**: Rule-based keyword matching
- **Router v2**: Trained classifier using TF-IDF + LogisticRegression
- **Router v3**: Cost-aware routing with mode features

## Router v1: Rule-Based

No training required. Uses predefined rules:

```python
from mm_orch.routing.router_v1 import RouterV1

router = RouterV1()
workflow, confidence, candidates = router.route("What is Python?", {})
```

## Router v2: Classifier Training

### Step 1: Collect Execution Traces

Run workflows and collect traces:

```bash
python -m mm_orch.main "What is Python?" --trace-output data/traces/traces.jsonl
```

### Step 2: Train Classifier

```bash
python scripts/train_router_v2.py \
    --traces data/traces/traces.jsonl \
    --output models/router_v2
```

Training process:
1. Extracts questions and chosen workflows from traces
2. Vectorizes questions using TF-IDF
3. Trains LogisticRegression classifier
4. Saves vectorizer and model

### Step 3: Use Trained Router

```python
from mm_orch.routing.router_v2 import RouterV2

router = RouterV2(
    vectorizer_path="models/router_v2/vectorizer.pkl",
    clf_path="models/router_v2/classifier.pkl"
)

workflow, confidence, candidates = router.route("Teach me Python", {})
```

## Router v3: Cost-Aware Training

### Step 1: Collect Cost Statistics

Cost stats are automatically collected during execution:

```python
from mm_orch.observability.cost_stats import CostStatsManager

manager = CostStatsManager("data/costs.json")
# Stats updated automatically via tracer
```

### Step 2: Train Cost-Aware Router

```bash
python scripts/train_router_v3.py \
    --traces data/traces/traces.jsonl \
    --costs data/costs.json \
    --output models/router_v3 \
    --lambda-cost 0.1
```

Training includes:
1. Text features (TF-IDF)
2. Mode features (one-hot encoded)
3. Cost statistics for scoring
4. Best-reward labeling strategy

### Step 3: Use Cost-Aware Router

```python
from mm_orch.routing.router_v3 import RouterV3

router = RouterV3(
    vectorizer_path="models/router_v3/vectorizer.pkl",
    clf_path="models/router_v3/classifier.pkl",
    costs_path="data/costs.json"
)

state = {"question": "What is Python?", "meta": {"mode": "chat"}}
workflow, score, candidates = router.route(state["question"], state)
```

## Training Data Format

### Trace Format (JSONL)

```json
{
  "request_id": "req-001",
  "question": "What is Python?",
  "chosen_workflow": "search_qa",
  "router_version": "v1",
  "mode": "default",
  "steps": [...],
  "quality_signals": {"citation_count": 2},
  "cost_stats": {"latency": 1500.0},
  "success": true
}
```

### Cost Stats Format (JSON)

```json
{
  "search_qa": {
    "execution_count": 100,
    "avg_latency_ms": 1500.0,
    "avg_vram_mb": 250.0,
    "avg_model_loads": 2.0,
    "success_rate": 0.95
  }
}
```

## Evaluation

### Accuracy Metrics

```python
from sklearn.metrics import classification_report

# Load test data
test_traces = load_traces("data/test_traces.jsonl")
X_test = [t.question for t in test_traces]
y_test = [t.chosen_workflow for t in test_traces]

# Predict
y_pred = [router.route(q, {})[0] for q in X_test]

# Evaluate
print(classification_report(y_test, y_pred))
```

### Cost-Quality Tradeoff

```python
# Compare routers
for router_name, router in [("v1", router_v1), ("v2", router_v2), ("v3", router_v3)]:
    total_cost = 0
    total_quality = 0
    
    for trace in test_traces:
        workflow, _, _ = router.route(trace.question, {})
        cost = cost_stats[workflow].avg_latency_ms
        quality = trace.quality_signals["citation_count"]
        
        total_cost += cost
        total_quality += quality
    
    print(f"{router_name}: Cost={total_cost:.0f}ms, Quality={total_quality}")
```

## Best Practices

1. **Collect diverse traces**: Include all workflow types in training data
2. **Balance dataset**: Ensure each workflow has sufficient examples
3. **Tune lambda**: Adjust cost weight based on your priorities
4. **Monitor performance**: Track router accuracy and cost savings
5. **Retrain regularly**: Update models as usage patterns change

## Troubleshooting

### Low Accuracy

- Collect more training data
- Check for class imbalance
- Try different classifiers (SVM, RandomForest)
- Add more features (question length, keywords)

### High Cost

- Increase lambda_cost parameter
- Review cost statistics accuracy
- Consider workflow optimizations

### Mode Features Not Working

- Verify mode is set in State.meta
- Check one-hot encoding in training script
- Ensure mode features are included in prediction

## See Also

- [Trace Format Documentation](./phase_b_trace_format.md)
- [Quality Signals Guide](./phase_b_quality_signals.md)
- [Cost Tracking Guide](./phase_b_cost_tracking.md)
