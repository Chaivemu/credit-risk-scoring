import pandas as pd
import numpy as np
import os

def generate_credit_data(n_samples=50000, seed=42):
    np.random.seed(seed)
    
    applicant_id = np.arange(1, n_samples + 1)
    age = np.random.randint(21, 75, n_samples)
    
    # Income (lognormal for realistic right skew)
    income = np.random.lognormal(mean=11.1, sigma=0.6, size=n_samples).astype(int)
    income = np.clip(income, 20000, 250000)
    
    employment_length_years = np.random.randint(0, 40, n_samples)
    home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], p=[0.4, 0.1, 0.5], size=n_samples)
    
    loan_amount = np.random.randint(1000, 40000, n_samples)
    loan_purpose = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'other'], 
        p=[0.55, 0.25, 0.1, 0.1], 
        size=n_samples
    )
    
    loan_grade = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 
                                  p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.05, 0.05], 
                                  size=n_samples)
    
    # Assign interest rate based on loan grade
    grade_rates = {'A': 0.07, 'B': 0.10, 'C': 0.13, 'D': 0.16, 'E': 0.20, 'F': 0.24, 'G': 0.28}
    interest_rate = np.array([np.random.normal(grade_rates[g], 0.01) for g in loan_grade])
    interest_rate = np.clip(interest_rate, 0.05, 0.35)
    
    # DTI generally higher for riskier borrowers
    dti_base = np.random.normal(0.18, 0.08, n_samples)
    grade_dti_shift = {'A': -0.05, 'B': -0.02, 'C': 0.0, 'D': 0.02, 'E': 0.05, 'F': 0.08, 'G': 0.12}
    dti_shift = np.array([grade_dti_shift[g] for g in loan_grade])
    dti = np.clip(dti_base + dti_shift, 0.0, 0.6)
    
    credit_history_length_years = np.random.randint(1, 40, n_samples)
    num_open_accounts = np.random.poisson(lam=10, size=n_samples)
    num_derogatory_marks = np.random.poisson(lam=0.2, size=n_samples)
    # Add derogatory marks for lower grades
    for i in range(n_samples):
        if loan_grade[i] in ['E', 'F', 'G'] and np.random.rand() > 0.5:
            num_derogatory_marks[i] += np.random.randint(1, 4)
            
    revolving_balance = np.random.exponential(scale=15000, size=n_samples).astype(int)
    
    # Utilization related to DTI and Grade
    revolving_utilization_rate = np.random.beta(a=2, b=5, size=n_samples) + dti * 0.5 + dti_shift
    revolving_utilization_rate = np.clip(revolving_utilization_rate, 0.0, 1.2)
    
    # Calculate default probability based on realistic features
    # Base logits (target around 20% default rate, logit around -1.38)
    logits = -2.5
    
    # Higher DTI -> higher default
    logits += dti * 3.5
    
    # Higher utilization -> higher default
    logits += revolving_utilization_rate * 2.0
    
    # More derogatory marks -> higher default
    logits += num_derogatory_marks * 0.8
    
    # Lower grade -> higher default
    grade_penalties = {'A': -1.5, 'B': -0.5, 'C': 0.0, 'D': 0.5, 'E': 1.0, 'F': 1.5, 'G': 2.0}
    logits += np.array([grade_penalties[g] for g in loan_grade])
    
    # Lower income -> slightly higher default
    logits -= np.log(income / 50000) * 0.5
    
    # Sigmoid to get probability
    probs = 1 / (1 + np.exp(-logits))
    
    # Actual default outcomes
    default = np.random.binomial(1, probs)
    
    df = pd.DataFrame({
        'applicant_id': applicant_id,
        'age': age,
        'income': income,
        'employment_length_years': employment_length_years,
        'home_ownership': home_ownership,
        'loan_amount': loan_amount,
        'loan_purpose': loan_purpose,
        'interest_rate': interest_rate,
        'loan_grade': loan_grade,
        'dti': dti,
        'credit_history_length_years': credit_history_length_years,
        'num_open_accounts': num_open_accounts,
        'num_derogatory_marks': num_derogatory_marks,
        'revolving_balance': revolving_balance,
        'revolving_utilization_rate': revolving_utilization_rate,
        'default': default
    })
    
    print(f"Generated dataset with {n_samples} records.")
    print(f"Overall Default Rate: {df['default'].mean():.2%}")
    
    return df

if __name__ == "__main__":
    df = generate_credit_data()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credit_risk_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"credit_risk_dataset.csv created successfully at {output_path}")
