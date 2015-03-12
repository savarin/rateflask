import numpy as np
import pandas as pd
from datetime import datetime


def process_features(df_raw, restrict_date=True, current_loans=True):
    '''
    Restricts data to loans of grades A, B, C, and D of desired issue dates, 
    then processes data by filling missing values and map to numerical format.

    Heuristics on labels:
    loan_status - proceeds reinvested in investment of same risk-return profile
        https://www.lendingclub.com/info/demand-and-credit-profile.action
    
    Heuristics on features:
    emp_length - 10 if >10 yrs, average if n/a
    earliest_cr_line - converted to length of time b/w first credit line and 
        date of loan issuance, issuance date of loan if n/a
    revol_util - missing values filled in by average
    mths_since_last - missing values filled with -1
    '''
    grade_mask = df_raw['grade'].isin(['A', 'B', 'C', 'D'])
    term_mask = df_raw['term'].str.contains('36', na=False)
    
    if restrict_date:
        if current_loans:
            date_mask = (df_raw['issue_d'].str.contains('2012', na=False)) \
                         | (df_raw['issue_d'].str.contains('2013', na=False)) \
                         | (df_raw['issue_d'].str.contains('2014', na=False))
        else:
            date_mask = (df_raw['issue_d'].str.contains('2009', na=False)) \
                         | (df_raw['issue_d'].str.contains('2010', na=False)) \
                         | (df_raw['issue_d'].str.contains('2011', na=False))
    else:
        date_mask = [True] * df_raw.shape[0]

    df_raw = df_raw[grade_mask & term_mask & date_mask]

    df = df_raw[['id', 'grade', 'sub_grade', 'issue_d', 'loan_status', 'int_rate',
                 'loan_amnt', 'term', 
                 'emp_length', 'annual_inc', 'dti',
                 'fico_range_low', 'fico_range_high', 'earliest_cr_line',
                 'open_acc', 'total_acc', 'revol_bal', 'revol_util', 'inq_last_6mths', 
                 'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
                 'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog',
                 'purpose', 'home_ownership']]

    df['loan_status'] = df['loan_status'].map({'Fully Paid': 1., 'Current': 1., 
                                               'In Grace Period': 0.76, 
                                               'Late (16-30 days)': 0.49, 
                                               'Late (31-120 days)': 0.28, 
                                               'Default': 0.08,
                                               'Charged Off': 0.})

    df['int_rate'] = df['int_rate'].map(lambda x: float(str(x).strip('%')) / 100)

    df['term'] = df['term'].map(lambda x: int(str(x).strip(' months')))

    df['emp_length'] = df['emp_length'].map(lambda x: '0.5 years' if x == '< 1 year' else x)
    df['emp_length'] = df['emp_length'].map(lambda x: '10 years' if x == '10+ years' else x)
    df['emp_length'] = df['emp_length'].map(lambda x: '-1 years' if x == 'n/a' else x)
    df['emp_length'] = df['emp_length'].map(lambda x: float(x.strip(' years')))
    emp_length_mean = np.mean([x for x in df['emp_length'].values if x > 0])
    df['emp_length'] = df['emp_length'].map(lambda x: emp_length_mean if x < 0 else x)

    df['annual_inc'] = df['annual_inc'].map(lambda x: float(x) / 12)
    df.rename(columns={'annual_inc': 'monthly_inc'}, inplace=True)

    df['fico_range_low'] = (df['fico_range_low'] + df['fico_range_high']) / 2.
    df.rename(columns={'fico_range_low': 'fico'}, inplace=True)

    df['earliest_cr_line'] = df[['earliest_cr_line', 'issue_d']].apply(lambda x: 
                                x['issue_d'] if pd.isnull(x['earliest_cr_line']) 
                                             else x['earliest_cr_line'], axis=1)
    df['earliest_cr_line'] = df[['earliest_cr_line', 'issue_d']].apply(lambda x: \
                                (datetime.strptime(x['issue_d'], '%b-%Y') \
                                - datetime.strptime(x['earliest_cr_line'], '%b-%Y')).days / 30, axis=1)

    df['revol_util'] = df['revol_util'].map(lambda x: float(str(x).strip('%')) / 100)
    revol_util_mean = np.mean(df['revol_util'])
    df['revol_util'] = df['revol_util'].fillna(revol_util_mean)

    df.rename(columns={'collections_12_mths_ex_med': 'collect_12mths'}, inplace=True)

    df['last_delinq'] = df['mths_since_last_delinq'].map(lambda x: -1 if x == 'n/a' else x)
    df['last_record'] = df['mths_since_last_record'].map(lambda x: -1 if x == 'n/a' else x)
    df['last_derog'] = df['mths_since_last_major_derog'].map(lambda x: -1 if x == 'n/a' else x)

    df['purpose_debt'] = df['purpose'].map(lambda x: x == 'debt_consolidation').astype(int)
    df['purpose_credit'] = df['purpose'].map(lambda x: x == 'credit_card').astype(int)
    df['purpose_home'] = df['purpose'].map(lambda x: x == 'home_improvement').astype(int)
    df['purpose_other'] = df['purpose'].map(lambda x: x == 'other').astype(int)
    df['purpose_buy'] = df['purpose'].map(lambda x: x == 'major_purchase').astype(int)
    df['purpose_biz'] = df['purpose'].map(lambda x: x == 'small_business').astype(int)
    df['purpose_medic'] = df['purpose'].map(lambda x: x == 'medical').astype(int)
    df['purpose_car'] = df['purpose'].map(lambda x: x == 'car').astype(int)
    df['purpose_move'] = df['purpose'].map(lambda x: x == 'moving').astype(int)
    df['purpose_vac'] = df['purpose'].map(lambda x: x == 'vacation').astype(int)
    df['purpose_house'] = df['purpose'].map(lambda x: x == 'house').astype(int)
    df['purpose_wed'] = df['purpose'].map(lambda x: x == 'wedding').astype(int)
    df['purpose_energy'] = df['purpose'].map(lambda x: x == 'renewable_energy').astype(int)

    df['home_mortgage'] = df['home_ownership'].map(lambda x: x == 'MORTGAGE').astype(int)
    df['home_rent'] = df['home_ownership'].map(lambda x: x == 'RENT').astype(int)
    df['home_own'] = df['home_ownership'].map(lambda x: x == 'OWN').astype(int)
    df['home_other'] = df['home_ownership'].map(lambda x: x == 'OTHER').astype(int)
    df['home_none'] = df['home_ownership'].map(lambda x: x == 'NONE').astype(int)
    df['home_any'] = df['home_ownership'].map(lambda x: x == 'ANY').astype(int)

    df = df.drop((['fico_range_high', 'mths_since_last_delinq', 'mths_since_last_record', 
                   'mths_since_last_major_derog', 'purpose', 'home_ownership']), axis=1)

    return df