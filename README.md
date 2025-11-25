
# Loan Default Prediction

This project analyzes LendingClub loan data to understand the factors that contribute to loan default and repayment. It combines exploratory data analysis (EDA) and predictive modeling to uncover patterns, visualize relationships, and compare model performance.

## Project Structure

- `data/`: Raw and processed datasets
	- `accepted_2007_to_2018Q4.csv.gz`, `rejected_2007_to_2018Q4.csv.gz`: Raw data files
	- `accepted_2007_to_2018q4.csv/`, `rejected_2007_to_2018q4.csv/`: Unzipped CSVs
	- `scripts/`: Data cleaning and joining scripts
- `EDA/`: Exploratory Data Analysis scripts and saved visualizations
	- `distributions.py`: Generates feature distributions and outcome plots
	- `graphs/`: Output figures from EDA
- `csv/`: Feature and column documentation
	- `columns.txt`: List and description of all columns
	- `features.txt`: Selected features for modeling

## Data Cleaning & Feature Engineering

- Outliers are removed using IQR filtering for key numeric features (e.g., annual income, DTI)
- Features such as FICO score, annual income, DTI, loan grade, and term are analyzed for their impact on loan outcomes
- Categorical variables are encoded and missing values handled appropriately

## Exploratory Data Analysis (EDA) Insights

- **FICO Scores**: Higher FICO scores are strongly associated with fully paid loans, while lower scores are more common among defaults.
- **Interest Rate**: Defaulted loans tend to have higher interest rates, with a more normal distribution, while fully paid loans are skewed toward lower rates.
- **Annual Income**: Both defaulted and fully paid loans show similar income distributions after outlier removal, suggesting income alone is not a strong differentiator.
- **DTI (Debt-to-Income Ratio)**: Higher DTI ratios are slightly more prevalent among defaulted loans, but the difference is modest.
- **Loan Grade & Term**: Lower grades and longer terms (60 months) are more common in defaulted loans.
- **Loan Purpose**: Certain purposes (e.g., debt consolidation) dominate among defaults, as shown in the Pareto chart.
- **Feature Correlations**: FICO scores, interest rates, and DTI show meaningful correlations with loan outcome, guiding feature selection for modeling.


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
| Logistic Regression  | 0.68              | 0.31                 | —            | ~0.73    |
| LightGBM             | 0.683             | —                    | —            | 0.7308   |
| XGBoost (tuned)      | 0.624             | —                    | 0.446        | ~0.73    |

**Major Insights:**
- High recall is critical for default detection; all models prioritized this over precision.
- LightGBM and XGBoost (after tuning) performed similarly, with LightGBM slightly ahead in ROC AUC.
- Feature engineering (ratios, flags, buckets) and handling class imbalance were essential for model success.
- Accuracy is misleading due to class imbalance; recall, precision, F1, and ROC AUC are better metrics.

## Major Insights

- Credit risk is most strongly associated with FICO score, interest rate, loan grade, and term.
- Income and DTI, while important, are less predictive on their own.
- Feature engineering and outlier removal improved model performance.
- Ensemble models (Gradient Boosting) significantly outperform linear models for this task.