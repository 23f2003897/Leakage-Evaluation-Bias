# Data Leakage & Evaluation Bias in Credit Risk Modeling

## Overview
This project presents a systematic investigation of **data leakage and evaluation bias**
in machine learning pipelines, using a real-world **credit risk prediction** problem.

Rather than focusing on maximizing model performance, the goal is to show **how incorrect
methodology can silently inflate metrics**, and how fixing these issues reveals the true
difficulty of the task.

The project contrasts a **naïve (leaky) pipeline** with a **leakage-safe pipeline**
and evaluates both under **random** and **temporal** splits.

---

## Dataset
- **Source:** Home Credit Default Risk
- **Samples:** 307,511 loan applications
- **Target:** Loan default (binary)
- **Class imbalance:** ~8% defaulters, ~92% non-defaulters

This imbalance makes evaluation especially sensitive to metric choice and leakage.

---

## Project Structure

├── notebook-00-integrity-audit.ipynb
├── notebook-01-naive-leaky-pipeline.ipynb
├── leakage-safe-credit-risk-pipeline.ipynb
├── comparison-evaluation-bias-quantification-concl.ipynb
├── README.md
└── .gitignore

### Notebook Breakdown

### 1️ Notebook 00 — Integrity Audit
- Target distribution and imbalance analysis
- Missing-value patterns
- Correlation inspection
- Identification of potential leakage-risk features

---

### 2️ Notebook 01 — Naïve / Leaky Pipeline
Intentionally flawed pipeline demonstrating:
- Preprocessing before train–test split
- Feature selection on full data
- Random split ignoring temporal effects
- Accuracy-based evaluation on imbalanced data

**Outcome:**  
High accuracy and reasonable ROC-AUC, but near-zero recall for defaulters.

---

### 3️ Notebook 02 — Leakage-Safe Pipeline
Pipeline rebuilt using best practices:
- Train–test split before preprocessing
- Pipeline-based imputation and encoding
- Imbalance-aware metrics (ROC-AUC, recall)
- Temporal evaluation to simulate deployment

**Outcome:**  
Lower but honest performance, revealing true model limitations.

---

### 4️ Notebook 03 — Comparison & Final Conclusions
- Side-by-side comparison of leaky vs safe pipelines
- Quantification of evaluation bias
- Research-style conclusions and takeaways

---

## Key Results

| Pipeline | ROC-AUC | Recall (Defaulters) |
|--------|--------|--------------------|
| Naïve (Leaky) | 0.710 | ~0.000 |
| Leakage-Safe (Random Split) | 0.749 | 0.013 |
| Leakage-Safe (Temporal Split) | 0.737 | 0.018 |

> **Accuracy is intentionally omitted** due to severe class imbalance
> and its inability to reflect real-world usefulness.

---

## Key Insights
- Data leakage inflates confidence, not real predictive capability.
- Accuracy and even ROC-AUC can hide complete failure on minority classes.
- Removing leakage does not guarantee better scores — it guarantees **trustworthy scores**.
- Temporal evaluation exposes optimistic bias in random splits.
- Methodology matters more than model choice.

---

## Final Takeaway
> **A weaker but honest model is more valuable than a strong model built on leaked information.**

---

## Tools & Technologies
- Python
- pandas, NumPy
- scikit-learn
- Jupyter Notebook
- Git & GitHub

---

## Future Work
This project serves as the research foundation for a follow-up applied system:
**Leakage-Safe Credit Risk Scoring System**, focusing on threshold tuning,
cost-sensitive decisions, and deployment considerations.

---

## Author
**Khusbu Pandey**

