# Lab 01 — Baseline Performance

## Goal
Compare Logistic Regression and Random Forest.

## Method
I used a 70/30 data split and measured accuracy, training time, inference latency, model size, and memory. Python 3.11.8, scikit-learn 1.5.0.

## Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 94.74% | 93.57% |
| Training | 0.984 s | 0.160 s |
| Inference | 0.0434 ms | 2.1117 ms |
| Size | 1.03 KB | 284.07 KB |

## Deployment Fit
Both fit cloud and edge latency and size limits. Logistic Regression also meets TinyML size and latency limits; Random Forest is too large.

## Conclusions
1. Logistic Regression was more accurate.
2. Random Forest trained faster, but predicted slower.
3. Deployment depends on speed, size, and memory.
 