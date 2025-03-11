Credit Risk Prediction for Loan Approval
Project Overview
This project involves developing a machine learning model with 96% accuracy to assess credit risk for a Non-Banking Financial Company (NBFC). The system automates the credit scoring process by categorizing loans into four risk categories: Poor, Fair, Good, and Excellent. A real-time dashboard was also built using Streamlit for visualizing loan default probability predictions, helping significantly reduce financial risk.

Key Features
Automated Credit Scoring: Classification of loans into risk categories.

Streamlit Dashboard: Real-time predictions of loan default probability for operational ease.

Custom Features for Better Predictions:

Delinquency Ratio Formula: (delinquent_months * 100) / total_loan_months

python
df_train_1['delinquency_ratio'] = (df_train_1['delinquent_months'] * 100 / df_train_1['total_loan_months']).round(1)
df_test['delinquency_ratio'] = (df_test['delinquent_months'] * 100 / df_test['total_loan_months']).round(1)
Average Days Past Due (DPD) Per Delinquency Formula: total_dpd / delinquent_months (if delinquent_months ≠ 0, else 0)

python
df_train_1['avg_dpd_per_delinquency'] = np.where(
    df_train_1['delinquent_months'] != 0,
    (df_train_1['total_dpd'] / df_train_1['delinquent_months']).round(1),
    0
)

df_test['avg_dpd_per_delinquency'] = np.where(
    df_test['delinquent_months'] != 0,
    (df_test['total_dpd'] / df_test['delinquent_months']).round(1),
    0
)
Loan-to-Income Ratio Formula: loan_amount / income

python
df_train_1['loan_to_income'] = round(df_train_1['loan_amount'] / df_train_1['income'], 2)
Project Highlights
High Accuracy: The machine learning model achieved a remarkable accuracy of 96%.

New Feature Engineering: Added features such as delinquency ratio, average DPD per delinquency, and loan-to-income ratio for better credit assessment.

Risk Mitigation: Provides data-driven insights for reducing financial risk for NBFCs.

Dashboard
A Streamlit-powered dashboard visualizes loan default probability predictions in real-time. It enhances operational efficiency by providing easy access to predictions and analyses.

How to Use
Clone the repository:

bash
git clone <repository-url>
Install the required libraries:

bash
pip install -r requirements.txt
Run the Streamlit dashboard:

bash
streamlit run app.py
Tech Stack
Programming Language: Python

Libraries: NumPy, Pandas, Scikit-learn, Streamlit

Deployment: Streamlit Dashboard

Acknowledgments
This project was developed as part of a credit risk analysis system for NBFCs, focused on providing actionable insights and improving financial decision-making processes
