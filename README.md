# Transaction Compliance Monitor

Combines unsupervised anomaly detection (CBLOF) with rule-based compliance checks, explains each flag with SHAP, and writes an audit log for reviewers.

## Data
`creditcard_small.csv`: 1,020 card transactions, 20 labeled fraud (1.96%). Features f1-f28 are anonymized PCA components; Amount and Time are the only interpretable columns.

## What it does
1. **CBLOF** (PyOD; 12 clusters, contamination 0.03, alpha 0.9, beta 3, seed 42) scores every transaction on standardized features.
2. **Rule checks** on Amount: high value (>= 99th percentile, 898.27), round amount (>= 50 and a multiple of 10), zero amount.
3. **SHAP** explains each CBLOF flag through a Random Forest surrogate trained to reproduce CBLOF scores (out-of-bag R2 = 0.843).
4. **Audit log** (`audit_log.csv`): one row per flagged transaction with score, rule tags, queue, top SHAP reasons, model settings and run timestamp. Fraud labels are excluded, as a production log would not have them.

## Results
| Check | Flagged | Fraud among flagged |
|---|---|---|
| CBLOF | 31 | 14 of 20 (recall 70.0%, precision 45.2%) |
| High-value rule | 11 | 1 |
| Round-amount rule | 37 | 0 |
| Zero-amount rule | 4 | 0 |
| CBLOF + rules combined | 77 | 14 (precision 18.2%) |

Review tiers: CBLOF-only 26 flagged (13 fraud); CBLOF + rule 5 (1 fraud); rule-only 46 (0 fraud); not flagged 943 (6 fraud missed).

## Key finding and design decision
The Amount rules added 46 extra flags and no extra fraud, and a rule hit did not raise fraud likelihood. So the review queue is ranked by CBLOF score. Rule hits appear as reason tags, and rule-only transactions go to a separate low-priority SAMPLE queue.

## Limitations
- Small sample: with 20 fraud cases, a single case moves the percentages a lot.
- CBLOF is unsupervised and fit on the whole dataset; labels are used only for evaluation.
- f1-f28 are anonymized, so SHAP shows which component drove a flag (f1 dominates), not a business reason.
- Rules use Amount only. There is no card ID, so velocity checks are not possible.
- The surrogate approximates CBLOF (R2 0.84), so SHAP reasons are approximate.

## Run
```
pip install pyod==1.1.3 shap scikit-learn pandas
python monitor.py
```
