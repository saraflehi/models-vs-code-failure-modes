# models-vs-code-failure-modes
Compare failure behaviors of deterministic code vs ML on the same task

# Models Lie Differently Than Code

**A comparative study of failure modes in rule-based vs. machine learning systems**

## Overview

This project analyzes how **deterministic rule-based code** and **statistical ML models** fail differently on the same task: detecting suspicious log lines.

Rather than just comparing accuracy metrics, I focused on understanding *how* and *why* each approach breaks down.

## The Task: Suspicious Log Detection

Given server log lines, flag potentially problematic entries (errors, slow queries, suspicious access patterns, rate limiting, etc.).

## Two Approaches Compared

### 1. Heuristic (Rule-Based) Detector
- **How it works:** Hand-coded rules checking for known patterns (`ERROR`, `429`, `timeout`, etc.)
- **Output:** Risk score + explanation of which rules triggered
- **Characteristics:**
  - Deterministic: same input → same output every time
  - Fully transparent and interpretable
  - **Failure mode:** Brittle—misses anything not explicitly coded

### 2. Statistical (ML) Detector
- **How it works:** TF-IDF features + Logistic Regression trained on labeled examples
- **Output:** Probability score for "suspicious"
- **Characteristics:**
  - Learns patterns from data without manual encoding
  - Can generalize to unseen patterns
  - **Failure mode:** Can be confidently wrong, especially on rare/edge cases

## Methodology

**Dataset:** Custom-generated log lines with labeled suspicious/normal cases  
**Metrics tracked:**
- Precision, Recall, F1-score (classification performance)
- Confusion matrices (where each approach fails)
- Per-example comparison (rule score vs. model probability)

**Analysis outputs:**
- `reports/comparison.csv` – side-by-side predictions for every test case
- Visualizations of failure patterns and confidence distributions

## Key Findings

### Different Failure Patterns

**Rule-based failures:**
- Misses log lines with only weak signals (e.g., single occurrence of `429` without other red flags)
- Classic threshold problem: signals don't matter unless they cross a hardcoded threshold
- **Fails predictably** – easy to diagnose and fix by adding rules

**ML model failures:**
- Can pick up subtle correlations the rules miss (e.g., unusual word combinations)
- But also **overfits to training data** and misclassifies rare phrasing
- **Fails unpredictably** – confident predictions on edge cases it's never seen

### Example Edge Case
```
Log: "Request took 1.2s | user_agent: suspicious-scanner-v2"
```

- **Rules:** Low score (only catches "suspicious" keyword, doesn't weight timing)
- **Model:** High confidence suspicious (learned that timing + user agent correlate)
- **Reality:** Actually suspicious—model wins here
```
Log: "429 rate limit applied to IP 192.168.1.100"
```

- **Rules:** Medium score (one red flag, but context looks normal)
- **Model:** Low confidence (rare phrasing not in training data)
- **Reality:** Should be flagged—rules win here

## Engineering Takeaways

| Aspect | Rule-Based | ML-Based |
|--------|------------|----------|
| **When to use** | Known patterns, need explainability | Large pattern space, patterns evolve |
| **Failure style** | Misses unknown cases | Confidently wrong on rare inputs |
| **Debugging** | Easy (trace which rule failed) | Hard (model is black box) |
| **Maintenance** | High (manually update rules) | Medium (retrain periodically) |

### Production Recommendation: Hybrid Approach

The best real-world system would combine both:
1. **Rules for high-confidence red flags** – instant blocking, fully explainable
2. **ML model for broader detection** – catch evolving patterns, rank by suspiciousness
3. **Human review for uncertain cases** – where model and rules disagree

This mirrors how modern security systems actually work (e.g., fraud detection, spam filters).

## Technologies Used

- **Python** – core implementation
- **Scikit-learn** – TfidfVectorizer, LogisticRegression, metrics
- **Pandas** – data processing and analysis
- **Matplotlib/Seaborn** – visualization (confusion matrices, score distributions)

## Project Structure
```
models-lie-differently/
├── data/
│   ├── generate_logs.py      # Synthetic log generation
│   └── logs.csv               # Generated dataset
├── detectors/
│   ├── heuristic.py           # Rule-based detector
│   └── ml_model.py            # ML detector (TF-IDF + LogReg)
├── evaluate.py                # Run both detectors and compare
├── visualize.py               # Generate plots and reports
├── reports/
│   ├── comparison.csv         # Side-by-side predictions
│   ├── confusion_matrices.png
│   └── score_distributions.png
└── README.md
```

## Running the Project
```bash
# Generate synthetic log data
python data/generate_logs.py

# Train and evaluate both detectors
python evaluate.py

# Generate visualizations
python visualize.py
```

## What I Learned

- **Software engineering isn't just about accuracy** – understanding failure modes matters more in production
- **No single approach is always better** – context determines the right tool
- **Interpretability vs. flexibility is a real trade-off** – you can't always have both
- **Hybrid systems are common in industry** – combining strengths of multiple approaches

## Future Improvements

- [ ] Test on real server logs (with privacy considerations)
- [ ] Add more ML models (Random Forest, Neural Network) for comparison
- [ ] Implement the hybrid approach and measure improvement
- [ ] Add uncertainty quantification (confidence intervals for ML predictions)
- [ ] Explore active learning (which uncertain cases to label next)

---

**License:** MIT  
**Contact:** Feel free to open an issue or reach out if you have questions!
