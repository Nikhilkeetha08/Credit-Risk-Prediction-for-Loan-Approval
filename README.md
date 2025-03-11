# Credit Risk Prediction for Loan Approval

This project develops a **machine learning model** with **96% accuracy** to assess credit risk for an NBFC. The model automates the credit scoring system, categorizing loans into **Poor, Fair, Good, and Excellent** risk levels. Additionally, a **Streamlit dashboard** enables real-time loan default probability predictions, significantly reducing financial risk.

## Features Engineered
1. **Delinquency Ratio:** Measures the percentage of delinquent months in total loan months.
   ```python
   df['delinquency_ratio'] = (df['delinquent_months'] * 100 / df['total_loan_months']).round(1)
   ```
2. **Average Days Past Due (DPD) per Delinquency:** Calculates the average delay in payments for delinquent months.
   ```python
   df['avg_dpd_per_delinquency'] = np.where(
       df['delinquent_months'] != 0, (df['total_dpd'] / df['delinquent_months']).round(1), 0)
   ```
3. **Loan-to-Income Ratio:** Represents the loan amount relative to the borrower's income.
   ```python
   df['loan_to_income'] = round(df['loan_amount'] / df['income'], 2)
   ```

## Technologies Used
- **Python** (pandas, numpy, scikit-learn, pickle)
- **Machine Learning** (Random Forest Classifier)
- **Streamlit** (for real-time predictions)

## How to Run the Project
1. Clone this repository:
   ```sh
   git clone https://github.com/Nikhilkeetha08/Credit-Risk-Prediction.git
   ```
2. Navigate to the project folder:
   ```sh
   cd Credit-Risk-Prediction
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Model Training & Accuracy
The model is trained using **Random Forest Classifier**, achieving **96% accuracy** on the test set.

## Streamlit Dashboard
The interactive **Streamlit** dashboard allows users to input loan details and get real-time predictions of credit risk.


