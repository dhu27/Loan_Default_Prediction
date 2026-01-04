
# Loan Default Prediction

This project analyzes LendingClub loan data to understand the factors that contribute to loan default and repayment. It combines exploratory data analysis (EDA) and predictive modeling to uncover patterns, visualize relationships, and compare model performance.

## Project Structure

- `EDA.ipynb`: Exploratory data analysis notebook
- `ML/`: Modeling notebooks
	- `LogisticRegression.ipynb`
	- `LightGBM.ipynb`
	- `XGboostFinal.ipynb`
- `scripts/`: Data cleaning / preprocessing scripts
	- `cleaning.py`: Generates the cleaned modeling dataset (`csv/filtered.csv`)
- `csv/`: CSVs and feature/column documentation
	- `accepted_2007_to_2018Q4.csv`, `rejected_2007_to_2018Q4.csv`: Source CSVs used by scripts
	- `filtered.csv`: Cleaned dataset used for EDA/modeling
	- `columns.txt`: List and description of all columns
	- `features.txt`: Selected features for modeling
- `data/`: Raw data storage (unzipped copies)
	- `accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`
	- `rejected_2007_to_2018q4.csv/rejected_2007_to_2018Q4.csv`

## Data Cleaning & Feature Engineering

- Outliers are removed using IQR filtering for key numeric features (e.g., annual income, DTI)
- Features such as FICO score, annual income, DTI, loan grade, and term are analyzed for their impact on loan outcomes
- Categorical variables are encoded and missing values handled appropriately

## Exploratory Data Analysis (EDA) Insights

- **FICO Scores**: The FICO low/high distributions appear essentially identical between defaulted and fully paid loans in this dataset, suggesting FICO alone is not a strong differentiator here.
- **Interest Rate**: Defaulted loans tend to have higher interest rates, with a more normal distribution, while fully paid loans are skewed toward lower rates.
- **Annual Income**: Both defaulted and fully paid loans show similar income distributions after outlier removal, suggesting income alone is not a strong differentiator.
- **DTI (Debt-to-Income Ratio)**: Higher DTI ratios are slightly more prevalent among defaulted loans, but the difference is modest.
- **Loan Grade & Term**: Lower grades and longer terms (60 months) are more common in defaulted loans.
- **Loan Purpose**: Certain purposes (e.g., debt consolidation) dominate among defaults, as shown in the Pareto chart.
- **Feature Correlations**: FICO scores, interest rates, and DTI show meaningful correlations with loan outcome, guiding feature selection for modeling.

## Power BI Dashboard

In addition to the notebooks, this project includes a comprehensive Power BI dashboard built to make the EDA + modeling outcomes easy to explore and present.

Typical views included in the dashboard:
- **Portfolio overview**: total loans, default rate, and volume by time
- **Risk segmentation**: default rate by grade, term, purpose, and other key categorical features
- **Borrower attributes**: distributions and default-rate lift by FICO, interest rate, DTI, and income bands
- **Model results (summary)**: side-by-side comparison of recall/precision/F1/AUC where available

Notes:
- The Power BI `.pbix` file is not currently tracked in this repository. If you want it included, add it under a folder like `powerbi/` and link it here.
- The dashboard is designed to use `csv/filtered.csv` as its primary data source.


## Modeling & Results

Three models were trained and compared: Logistic Regression, LightGBM, and XGBoost. Each model used engineered features and handled class imbalance.

### Logistic Regression
- **Recall (defaults):** 0.68
- **Precision (defaults):** 0.31
- **Insight:** The model identifies most defaults (high recall), but with lower precision, meaning more false positives. This is acceptable in credit risk, where missing a default is costly.

### LightGBM
- **Recall (defaults):** 0.683
- **ROC AUC:** 0.7308
- **Insight:** LightGBM performed similarly to logistic regression in recall, but with a slightly higher ROC AUC, indicating better overall discrimination. It efficiently handled many features and class imbalance.

### XGBoost
- **Initial Recall (defaults):** 0.089 (very conservative, only high-probability defaults)
- **Tuned Recall (defaults):** 0.624
- **Tuned F1 (defaults):** 0.446
- **Insight:** XGBoost required threshold tuning to achieve competitive recall. After tuning, it correctly identified about 62% of defaults, with a balanced F1 score. Accuracy was less meaningful due to class imbalance.

### Model Comparison Table

| Model                | Recall (defaults) | Precision (defaults) | F1 (defaults) | ROC AUC  |
|----------------------|-------------------|----------------------|--------------|----------|
| Logistic Regression  | 0.680             | 0.310                | 0.420        | 0.7062   |
| LightGBM             | 0.683             | 0.330                | 0.445        | 0.7308   |
| XGBoost (tuned)      | 0.624             | 0.347                | 0.446        | 0.7296   |

Table notes:
- Logistic Regression metrics come from the saved `classification_report` output in `ML/LogisticRegression.ipynb`.
- XGBoost tuned-threshold metrics come from the tuned-threshold `classification_report` output in `ML/XGboostFinal.ipynb`. AUC is reported on the predicted probabilities.
- LightGBM metrics come from the `classification_report` and printed ROC AUC output in `ML/LightGBM.ipynb`.

**Major Insights:**
- High recall is critical for default detection; all models prioritized this over precision.
- LightGBM and XGBoost (after tuning) performed similarly, with LightGBM slightly ahead in ROC AUC.
- Feature engineering (ratios, flags, buckets) and handling class imbalance were essential for model success.
- Accuracy is misleading due to class imbalance; recall, precision, F1, and ROC AUC are better metrics.

## Major Insights

- Credit risk is most strongly associated with FICO score, interest rate, loan grade, and term.
- Income and DTI, while important, are less predictive on their own.
- Feature engineering and outlier removal improved model performance.
- Ensemble models (Gradient Boosting) modestly outperform the linear baseline for this task.
