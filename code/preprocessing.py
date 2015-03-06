
import pickle
import numpy as np
import pandas as pd
from datetime import datetime


def dump_to_pickle(file_object, file_path):
    pickle.dump(file_object, open(file_path, 'w'))


def load_from_pickle(file_path):
    return pickle.load(open(file_path, 'r'))


def process_features(df_raw, current_loans=True):
    '''
    Restricts data to loans of grades A, B, C, and D of desired issue dates, 
    then processes data by filling missing values and map to numerical format.

    Heuristics on features:
    loan_status - proceeds reinvested in investment of same risk-return profile
    https://www.lendingclub.com/info/demand-and-credit-profile.action

    emp_length - 10 if >10 yrs, average if n/a
    earliest_cr_line - converted to length of time b/w first credit line and 
        date of loan issuance, issuance date of loan if n/a
    revol_util - missing values filled in by average
    '''
    grade_mask = df_raw['grade'].isin(['A', 'B', 'C', 'D'])
    term_mask = df_raw['term'].str.contains('36', na=False)
    
    if current_loans:
        date_mask = (df_raw['issue_d'].str.contains('2012', na=False)) \
                     | (df_raw['issue_d'].str.contains('2013', na=False)) \
                     | (df_raw['issue_d'].str.contains('2014', na=False))
    else:
        date_mask = (df_raw['issue_d'].str.contains('2009', na=False)) \
                     | (df_raw['issue_d'].str.contains('2010', na=False)) \
                     | (df_raw['issue_d'].str.contains('2011', na=False))


    df_raw = df_raw[grade_mask & term_mask & date_mask]

    df = df_raw[['id', 'grade', 'sub_grade', 'issue_d', 'loan_status', 'int_rate',
                 'loan_amnt', 'term', 
                 'emp_length', 'annual_inc', 'dti',
                 'fico_range_low', 'fico_range_high', 'earliest_cr_line',
                 'open_acc', 'total_acc', 'revol_bal', 'revol_util', 'inq_last_6mths', 
                 'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
                 'purpose', 'home_ownership']]

    df['loan_status'] = df['loan_status'].map({'Fully Paid': 1., 'Current': 1., 
                                               'In Grace Period': 0.76, 
                                               'Late (16-30 days)': 0.49, 
                                               'Late (31-120 days)': 0.28, 
                                               'Default': 0.08,
                                               'Charged Off': 0.})

    # Rows with missing interest rates appear to be corrupted, only in 3a
    df = df[pd.notnull(df['int_rate'])]
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

    df['purpose'] = df['purpose'].map({'credit_card':'credit', 'debt_consolidation':'debt', 
                                       'home_improvement':'home', 'major_purchase':'buy', 
                                       'medical':'medic', 'renewable_energy':'energy', 
                                       'small_business':'biz', 'vacation':'vac', 
                                       'wedding':'wed'})

    df = pd.concat([df, pd.get_dummies(df['purpose'], prefix='purpose')], axis=1)

    df['home_ownership'] = df['home_ownership'].map(lambda x: x.lower())
    df.rename(columns={'home_ownership': 'home_own'}, inplace=True)
    df = pd.concat([df, pd.get_dummies(df['home_own'], prefix='home_own')], axis=1)

    df = df.drop((['fico_range_high', 'purpose', 'home_own']), axis=1)

    return df


def process_payment(df_raw):
    '''
    Process payment information pertaining to matured loans.
    '''
    grade_mask = df_raw['grade'].isin(['A', 'B', 'C', 'D'])
    term_mask = df_raw['term'].str.contains('36', na=False)
    date_mask = (df_raw['issue_d'].str.contains('2009', na=False)) \
                | (df_raw['issue_d'].str.contains('2010', na=False)) \
                | (df_raw['issue_d'].str.contains('2011', na=False))

    df_raw = df_raw[grade_mask & term_mask & date_mask]

    df = df_raw[['id', 'sub_grade', 'installment']]
    df['int_rate'] = df_raw['int_rate'].map(lambda x: float(str(x).strip('%')) / 100).values

    # Residual amount for current loans and those in grace period under
    # consideration are sufficiently small and considered fully paid
    df['default_status'] = df_raw['loan_status'].map({
                                        'Fully Paid': 0, 
                                        'Does not meet the credit policy.  Status:Fully Paid': 0, 
                                        'Current': 0,
                                        'In Grace Period': 0,
                                        'Charged Off': 1, 
                                        'Does not meet the credit policy.  Status:Charged Off': 1,
                                        'Late (31-120 days)': 1})

    df['total_rec_pymnt'] = df_raw['total_rec_prncp'] + df_raw['total_rec_int']
    df['total_at_maturity'] = df_raw['total_rec_late_fee'] + df_raw['recoveries'] \
                                     - df_raw['collection_recovery_fee']

    # Calculates the number of months installment fully paid
    df['months_paid'] = df[['total_rec_pymnt', 'installment']] \
                                .apply(lambda x: np.floor(x['total_rec_pymnt'] \
                                                 / x['installment']), axis=1)

    # Calculates the fraction of installment paid on final month
    df['residual'] = df[['total_rec_pymnt', 'installment', 'months_paid']] \
                            .apply(lambda x: (x['total_rec_pymnt'] / x['installment']) \
                                             - x['months_paid'], axis=1)

    # Calculates the amount received from recovery, assumed to be at maturity
    # and quoted as a multiple of the monthly installment
    df['recovery'] = df[['total_at_maturity', 'installment']] \
                            .apply(lambda x: (x['total_at_maturity'] \
                                              / x['installment']), axis=1)

    return df[['sub_grade', 'int_rate', 
               'default_status', 'months_paid', 'residual', 'recovery']]


def convert_to_array(df,
                     date_range,
                     features,
                     create_label=False,
                     label='sub_grade',
                     label_one='A1',
                     label_zero='D5'):
    '''
    Converts dataframe to array, with option to create target column
    '''
    date_mask = df['issue_d'].isin(date_range)

    if create_label:
        label_mask = df[label].isin([label_one, label_zero])
        X = df[date_mask & label_mask][features].values
        y = df[date_mask & label_mask][label].map({label_one:1, label_zero:0}).values 
    else:
        X = df[date_mask][features].values
        y = df[date_mask][label].values 

    return X, y