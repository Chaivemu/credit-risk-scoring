# Credit Risk Scoring Model

> End-to-end credit risk scoring pipeline — EDA, feature engineering, XGBoost modeling, and FICO-style scorecard generation on 50K synthetic loan applicants.

---

## Project Overview

This project presents a complete, graduate-level credit risk scoring solution. It walks through building a synthetic dataset representative of a real-world lending environment, exploring the data, engineering features (including Weight of Evidence encoding), training multiple ML algorithms, and converting predictive output into a standard FICO-style points scorecard.

---

## Files & Deliverables

| File | Description |
|------|-------------|
| `credit_risk_dataset.csv` | 50,000 synthetically generated loan applicants with realistic correlations between DTI, utilization, loan grade, and default |
| `credit_risk_analysis.ipynb` | Full data science pipeline — EDA, model comparisons, evaluation, and scorecard generation |
| `customer_score_report.csv` | Final output scoring each applicant 300–850 with risk bands and recommended actions |
| `generate_data.py` | Data generation script |
| `generate_notebook.py` | Notebook generator script |
| `requirements.txt` | Python package dependencies |

---

## Model Performance

XGBoost was the strongest overall performer, offering an optimal balance of discrimination and stability.

| Model | ROC-AUC | Gini | KS Value | Brier Score |
|-------|---------|------|----------|-------------|
| **XGBoost** | ~0.82 | ~0.64 | ~0.48 | ~0.16 |
| LightGBM | ~0.81 | ~0.62 | ~0.47 | ~0.16 |
| Random Forest | ~0.79 | ~0.58 | ~0.44 | ~0.17 |
| Logistic Regression | ~0.76 | ~0.52 | ~0.40 | ~0.18 |
| Decision Tree | ~0.72 | ~0.44 | ~0.35 | ~0.19 |

> Exact values vary slightly by random seed but the hierarchy remains consistent. Refer to notebook outputs for precise results.

---

## Key Business Findings

1. **DTI + Utilization Interaction** — The combination of high Debt-to-Income (>35%) and elevated Revolving Utilization (>80%) is the strongest precursor to default.
2. **Income Relative to Loan Size** — Debt consolidation loans taken by borrowers with a low Income-to-Loan ratio result in higher predicted exposure losses compared to standard credit card refinancing.
3. **Actionable Score Bands** — By converting probability to a 300–850 scoring system, underwriters can apply clear threshold criteria and limit manual review to the 580–700 margin, reducing operational cost.

### Recommended Actions by Score Band

| Score Band | Score Range | Action |
|------------|-------------|--------|
| Very Low Risk | 750 – 850 | Auto-Approve |
| Low Risk | 700 – 749 | Auto-Approve |
| Medium Risk | 650 – 699 | Manual Review |
| High Risk | 580 – 649 | Decline |
| Very High Risk | 300 – 579 | Auto-Decline |

---

## How to Run

### 1. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Open the notebook

```bash
jupyter notebook credit_risk_analysis.ipynb
```

### (Optional) Rebuild from scratch

```bash
# Step 1 — Regenerate dataset
python generate_data.py

# Step 2 — Recompile notebook
python generate_notebook.py

# Step 3 — Execute notebook end-to-end
jupyter nbconvert --execute --to notebook --inplace credit_risk_analysis.ipynb
```

---

## Pipeline Overview

```
Raw Data Generation
       ↓
Exploratory Data Analysis (EDA)
       ↓
Feature Engineering (WoE, Binning, Interactions)
       ↓
Model Training (LogReg, DT, RF, XGBoost, LightGBM)
       ↓
Evaluation (ROC-AUC, Gini, KS, Brier, SHAP)
       ↓
FICO-Style Scorecard (300–850)
       ↓
Risk Band Assignment + Business Recommendations
```
