import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('csv/filtered.csv')

#overall outcomes

sns.countplot(data=df, y = "outcome", hue = "outcome", palette = "Set1")
plt.title("Loan Outcomes")
plt.yticks([0, 1], ["Default", "Fully Paid"])
plt.ylabel("Outcome")
plt.xlabel("Frequency (1e6)") 
plt.savefig('EDA/graphs/loan_outcomes.png')
plt.show()

#distribution of fico credit scores based on outcomes

fig, axes= plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=df, x='fico_range_low', hue='outcome', ax=axes[0])
axes[0].set_title("FICO Range Low Distribution by Loan Default")
axes[0].set_xlabel("FICO Range Low")
axes[0].set_ylabel("Frequency")

sns.histplot(data=df, x='fico_range_high', hue='outcome', ax=axes[1])
axes[1].set_title("FICO Range High Distribution by Loan Default")
axes[1].set_xlabel("FICO Range High")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
fig.savefig('EDA/graphs/fig_fico_range_low_high.png')
plt.show()

#Loan Grade vs Outcome

ax = sns.countplot(data=df, y = "outcome", 
              hue = "grade", hue_order = sorted(df['grade'].unique()),
              palette = "Set2")
plt.title("Loan Outcomes")
plt.yticks([0, 1], ["Default", "Fully Paid"])
plt.ylabel("Outcome")
plt.xlabel("Frequency") 

ax.legend(title = "Grade")

plt.savefig('EDA/graphs/loan_grade_vs_outcome.png')
plt.show()

#fully paid seems to be more skewed right (despite grade being a categorical variable)
#defaulted loans have a more normal distribution

#Loan Term vs Outcome
ax = sns.countplot(data = df, y = "outcome", 
              hue = "term", 
              palette = "Set1")

plt.ylabel("Loan Term")
plt.xlabel("Frequency")
plt.yticks([0, 1], ["Default", "Fully Paid"])

ax.legend(title = "Loan Term")

plt.savefig('EDA/graphs/loan_term_vs_outcome.png')
plt.show()

# 60 month term loans made up a much larger portion of the defaulted loans than the fully paid loans

# Interest rate distribution by outcome
default_df = df[df['outcome'] == 0]
fully_df = df[df['outcome'] == 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data = default_df, x = "int_rate", bins = 20, ax = axes[0], color = "red")
axes[0].set_title("Interest Rate Distribution for Defaulted Loans")
axes[0].set_xlabel("Interest Rate")


sns.histplot(data = fully_df, x = "int_rate", bins = 20, ax = axes[1], color = "lightgreen")
axes[1].set_title("Interest Rate Distribution for Fully Paid Loans")
axes[1].set_xlabel("Interest Rate")

#Defaulted loans tend to have a more normal distribution that's slightly skewed high, but fully paid loans are definitively skewed high
fig.savefig('EDA/graphs/interest_rate_distribution_by_outcome.png')
#Defaulted loans tend to have a more normal distribution that's slightly skewed high, but fully paid loans are definitively skewed high

def remove_outliers_iqr(df, cols, k=1.5):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

#remove outliers from selected numeric features
numeric_cols = ["annual_inc"]
df_clean = remove_outliers_iqr(df, numeric_cols)

#annual income vs outcome 
default_df = df_clean[df_clean['outcome'] == 0]
fully_df = df_clean[df_clean['outcome'] == 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data = default_df, x = "annual_inc", bins = 20, ax = axes[0], color = "red")
axes[0].set_title("Annual Income Distribution for Defaulted Loans")
axes[0].set_xlabel("Annual Income")
axes[0].tick_params(axis='x', rotation=45)



sns.histplot(data = fully_df, x = "annual_inc", bins = 20, ax = axes[1], color = "lightgreen")
axes[1].set_title("Annual Income Distribution for Fully Paid Loans")
axes[1].set_xlabel("Annual Income")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
fig.savefig('EDA/graphs/annual_income_distribution_by_outcome.png')
plt.show()
#nearly identical distributions


#remove outliers
numeric_cols = ["dti"]
df_clean = remove_outliers_iqr(df, numeric_cols)

default_df = df_clean[df_clean['outcome'] == 0]
fully_df = df_clean[df_clean['outcome'] == 1]

#dti ratio vs outcome
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Boxplot for defaulted loans
sns.boxplot(data=default_df, y="dti", color="red", ax=axes[0])
axes[0].set_title("DTI Distribution for Defaulted Loans")
axes[0].set_ylabel("DTI Ratio")
axes[0].set_xlabel("Defaulted")

# Boxplot for fully paid loans
sns.boxplot(data=fully_df, y="dti", color="lightgreen", ax=axes[1])
axes[1].set_title("DTI Distribution for Fully Paid Loans")
axes[1].set_ylabel("DTI Ratio")
axes[1].set_xlabel("Fully Paid")

plt.tight_layout()
fig.savefig('EDA/graphs/dti_distribution_by_outcome.png')
plt.show()


#nearly identical distributions when excluding outliers

#loan purpose pareto

# Subset: defaulted loans only
default_df = df[df['outcome'] == 0]

# Count loan purposes
purpose_counts = default_df['purpose'].value_counts().reset_index()
purpose_counts.columns = ['purpose', 'count']

# Cumulative percentage
purpose_counts['cumperc'] = purpose_counts['count'].cumsum() / purpose_counts['count'].sum() * 100

# Plot
fig, ax1 = plt.subplots(figsize=(12,6))

# Bars (Pareto part)
sns.barplot(
    data=purpose_counts,
    x="purpose", 
    y="count", 
    ax=ax1, 
    palette="Set2", 
    order=purpose_counts["purpose"]  # ensures Pareto order
)

ax1.set_ylabel("Frequency")
ax1.set_xlabel("Loan Purpose")
ax1.tick_params(axis='x', rotation=45)
ax1.set_title("Pareto Chart of Loan Purposes (Defaults)")

plt.tight_layout()
fig.savefig('EDA/graphs/pareto_loan_purpose_defaults.png')
plt.show()

#Defaulted loans tend to have a more normal distribution that's slightly skewed high, but fully paid loans are definitively skewed high

#Large Heatmap
#heat map
heatmap_vars = [
    "loan_amnt","int_rate","installment",
    "annual_inc","dti",
    "fico_range_low","fico_range_high",
    "inq_last_6mths","open_acc","total_acc",
    "revol_bal","revol_util","tot_cur_bal","total_rev_hi_lim",
    "mort_acc","avg_cur_bal","bc_util","pub_rec_bankruptcies"
]

plt.figure(figsize=(14,10))
sns.heatmap(df[heatmap_vars].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Key Features")
plt.savefig('EDA/graphs/correlation_heatmap_key_features.png')
plt.show()

#Core Feature Heatmap
#core head map
core_vars = [
    "loan_amnt","int_rate","installment",
    "annual_inc","dti",
    "fico_range_low","fico_range_high",
    "revol_util","open_acc","total_acc"
]

plt.figure(figsize=(10,8))
sns.heatmap(df[core_vars].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Core 10 Features)")
plt.savefig('EDA/graphs/correlation_heatmap_core_features.png')
plt.show()

