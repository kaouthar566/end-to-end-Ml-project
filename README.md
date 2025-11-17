# *end-to-end-Ml-project*
## Loan-default-prediction
### context
#### Retail banks and financial institutions face significant challenges in managing credit risk. A major portion of their revenue comes from interest earned on loans, particularly home loans and personal loans. However, defaults on these loans, also known as non-performing loans (NPLs), can severely impact profitability and liquidity. Traditional credit assessment methods often rely on manual evaluation and historical credit scoring, which may be time-consuming, error-prone, and unable to capture complex patterns in borrower behavior. With the increasing availability of digital financial data, there is an opportunity to leverage machine learning models to improve loan risk assessment and decision-making.
### Problem
#### Banks struggle with identifying potential defaulters accurately before loan disbursement. Manual assessments and rule-based systems can result in:
High rates of loan defaults due to overlooked risk factors. Inefficient resource allocation, as low-risk applicants may be unnecessarily rejected or high-risk applicants approved. Financial losses and regulatory challenges due to inaccurate risk prediction. The core problem is the lack of an effective, data-driven system to predict the likelihood of loan default.
### Solution
#### The proposed solution is to develop a Loan Default Prediction model using machine learning. This model will: Analyze historical loan data and borrower profiles to identify patterns associated with defaults. Provide risk scores or classifications for each applicant to support informed lending decisions. Reduce financial losses and improve portfolio quality by flagging high-risk borrowers early. Enhance efficiency by automating part of the credit assessment process, reducing human error and bias.
### Data
#### 
- Source: HMEQ / Kaggle
- Size: 5,960 registers
- Features: 13 feautures
- Target: Binary (0 = No Default, 1 = Default)
### Features Description 
● BAD: 1 = Client defaulted on loan, 0 = loan repaid

● LOAN: Amount of loan approved

● MORTDUE: Amount due on the existing mortgage

● VALUE: Current value of the property

● REASON: Reason for the loan request (HomeImp = home improvement, DebtCon= debt consolidation which means taking out a new loan to pay off other liabilities and consumer debts)

● JOB: The type of job that loan applicant has such as manager, self, etc.

● YOJ: Years at present job

● DEROG: Number of major derogatory reports (which indicates serious delinquency or late payments).

● DELINQ: Number of delinquent credit lines (a line of credit becomes delinquent when a borrower does not make the minimum required payments 30 to 60 days past the day on which the payments were due)

● CLAGE: Age of the oldest credit line in months

● NINQ: Number of recent credit inquiries

● CLNO: Number of existing credit lines

● DEBTINC: Debt-to-income ratio (all monthly debt payments divided by gross monthly income. This number is one of the ways lenders measure a borrower’s ability to manage the monthly payments to repay the money they plan to borrow)


