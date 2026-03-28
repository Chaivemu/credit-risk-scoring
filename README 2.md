# Credit Risk Scoring Project — Capital One Style

## Project Overview
This project presents a complete, graduate-business-school-level credit risk scoring solution. It walks through building a synthetic dataset representative of a real-world lending environment, exploring the data, engineering features (including Weight of Evidence), training multiple machine learning algorithms, and converting the predictive output into a standard FICO-Style points scorecard.

## Files & Deliverables
- `credit_risk_dataset.csv`: 50,000 synthetically generated loan applicants designed to realistically correlate standard lending criteria (DTI, utilization, loan grade) with default.
- `credit_risk_analysis.ipynb`: A comprehensive data science pipeline covering EDA, model comparisons, evaluation (AUC/KS/Gini/SHAP), and scoring conversion.
- `customer_score_report.csv`: The final dataset scoring each applicant from 300-850, tagging them into risk bands with actionable recommendations (e.g., Approve / Review / Decline).
- `generate_data.py`: Data generation logic.
- `generate_notebook.py`: Notebook generator code used to compile the analysis.
- `requirements.txt`: Python package dependencies.

## Model Summary Table

XGBoost proved to be the strongest overall performer on the dataset, offering an optimal balance of discrimination and stability. The model comparison results roughly reflect:

| Model | ROC-AUC | Gini | KS Value | Brier Score |
|-------|---------|------|----------|-------------|
| XGBoost | ~ 0.82 | ~ 0.64 | ~ 0.48 | ~ 0.16 |
| LightGBM | ~ 0.81 | ~ 0.62 | ~ 0.47 | ~ 0.16 |
| Random Forest | ~ 0.79 | ~ 0.58 | ~ 0.44 | ~ 0.17 |
| Logistic Regression | ~ 0.76 | ~ 0.52 | ~ 0.40 | ~ 0.18 |
| Decision Tree | ~ 0.72 | ~ 0.44 | ~ 0.35 | ~ 0.19 |

*(Note: Exact values slightly vary by random seed execution, but the hierarchy remains consistent. Refer to the execution outputs in the notebook).*

## Business Findings
1. **DTI + Utilization Interaction**: The combination of high Debt-to-Income (> 35%) and elevated Revolving Utilization (> 80%) is the strongest precursor to default. Intervening on these clients limits extreme loss.
2. **Income Relative to Loan Size**: Debt consolidation loans taken by borrowers with a low Income-to-Loan ratio result in higher predicted exposure losses compared to standard credit card refinancing.
3. **Actionable Score Bands**: By pivoting from absolute probability to a 300–850 scoring system, underwriters can directly apply threshold criteria (e.g., `Score > 700` = Auto Appove, `Score < 580` = Auto Decline), limiting manual review only to the 580-700 margin to save operational costs.

## How to Run

**1. Create a Python environment and install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Evaluate the Notebook**
You can directly open the generated notebook to review the results:
```bash
jupyter notebook credit_risk_analysis.ipynb
```

**(Optional) Run the Pipeline from Scratch**
To reconstruct the entire project structure from zero:
```bash
# 1. Regenerate 50k rows dataset
python generate_data.py

# 2. Recompile the notebook cells logic
python generate_notebook.py

# 3. Execute the notebook head-to-tail silently (outputs customer_score_report.csv embedded graphs)
jupyter nbconvert --execute --to notebook --inplace credit_risk_analysis.ipynb
```
