import pandas as pd
import numpy as np

df = pd.read_csv('csv/accepted_2007_to_2018Q4.csv')

filtered = df[df['loan_status'].notna()]
filtered = filtered[filtered['loan_status'].isin(['Fully Paid', 
                                                  'Does not meet the credit policy. Status:Fully Paid', 
                                                  'Charged Off',
                                                  'Default',
                                                  'Does not meet the credit policy. Status:Charged Off'])]


feature_map = {
    'Fully Paid': 1,
    'Does not meet the credit policy. Status:Fully Paid': 1,
    'Charged Off': 0,
    'Default': 0,
    'Does not meet the credit policy. Status:Charged Off': 0
}

filtered['outcome'] = filtered['loan_status'].map(feature_map)

# drop columns that are unnecessary identifers or metadata

filtered = filtered.drop(columns=['id', 'member_id', 'url', 'desc', 'title'])

#drop columns that aren't known at loan origination

filtered = filtered.drop(columns=['hardship_flag', 'hardship_type', 'hardship_reason', 
                                  'hardship_status', 'deferral_term', 'hardship_amount',
                                  'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
                                  'hardship_length', 'hardship_dpd', 'hardship_loan_status', 'orig_projected_additional_accrued_interest',
                                  'hardship_payoff_balance_amount', 'hardship_last_payment_amount'])

#drop columns that cause data leakage

filtered = filtered.drop(columns=['out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
                                  'total_rec_int', 'total_rec_late_fee', 'recoveries', 'issue_d',
                                  'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
                                  'last_fico_range_high', 'last_fico_range_low'])

#drop settlement-related post default outcomes

filtered = filtered.drop(columns=['debt_settlement_flag', 'debt_settlement_flag_date', 
                                  'settlement_status', 'settlement_date', 'settlement_amount', 
                                  'settlement_percentage', 'settlement_term'])


#messy columns that would require NLP
filtered = filtered.drop(columns=['emp_title'])

#dropped for low variation (causes noise)
filtered = filtered.drop(columns=['policy_code', 'pymnt_plan', 'disbursement_method'])


#print(filtered.shape)

#proportion of NAs
for column in filtered.columns:
    if filtered[column].isna().sum() != 0:
        missing = filtered[column].isna().sum()
        portion = (missing / filtered.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")


#filter columns w/ large portions of NAs

filtered = filtered.drop(columns=[
    # High missingness (50%+)
    'mths_since_last_delinq',      # ~50% missing
    'mths_since_last_record',      # ~83% missing
    'mths_since_last_major_derog', # ~74% missing
    'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', # ~60%
    'mths_since_rcnt_il',          # ~61%
    'total_bal_il',                # ~60%
    'il_util',                     # ~65%
    'open_rv_12m', 'open_rv_24m',  # ~60%
    'max_bal_bc',                  # ~60%
    'all_util',                    # ~60%
    'inq_fi',                      # ~60%
    'total_cu_tl',                 # ~60%
    'inq_last_12m',                # ~60%
    'mths_since_recent_bc_dlq',    # ~76%
    'mths_since_recent_revol_delinq', # ~67%

    # Extreme missingness (joint/secondary applicant fields, ~98–99%)
    'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'revol_bal_joint',
    'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
    'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util',
    'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
    'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog'
])

filtered.to_csv('csv/filtered.csv', index=False)
