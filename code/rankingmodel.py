import numpy as np
import pandas as pd
from datetime import datetime


def process_data(df_raw):
    '''
    Processes raw data - fills missing values and map to numerical format.

    Heuristics:
    loan_status - proceeds reinvested in investment with same risk-return profile
    https://www.lendingclub.com/info/demand-and-credit-profile.action
    
    emp_length - 10 if >10 yrs, average if n/a
    earliest_cr_line - converted to time b/w first credit line and date of loan issuance
    revol_util - missing values filled in by average
    '''
    df = df_raw[['id', 'grade', 'sub_grade', 'issue_d', 'loan_status', 'int_rate',
                 'loan_amnt', 'term', 
                 'emp_length', 'annual_inc', 'dti',
                 'fico_range_low', 'fico_range_high', 'earliest_cr_line',
                 'open_acc', 'total_acc', 'revol_bal', 'revol_util', 'inq_last_6mths', 
                 'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
                 'purpose', 'home_ownership']]

    df['loan_status'] = df['loan_status'].map({'Fully Paid':1., 'Current':1., 
                                           'In Grace Period':0.76, 
                                           'Late (16-30 days)':0.49, 
                                           'Late (31-120 days)':0.28, 
                                           'Default':0.08,
                                           'Charged Off':0.})

    df['int_rate'] = df['int_rate'].map(lambda x: float(str(x).strip('%')) / 100)

    df['term'] = df['term'].map(lambda x: int(x.strip(' months')))

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

    df['earliest_cr_line'] = df[['earliest_cr_line', 'issue_d']].apply(lambda x:\
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


def main():
    df_2014 = pd.read_csv('../data/loans3c.csv', header=True).iloc[:-2, :]
    df_2013 = pd.read_csv('../data/loans3b.csv', header=True).iloc[:-2, :]
    df_raw = pd.concat((df_2014, df_2013.iloc), axis=0)

    df = process_data(df_raw)


if __name__ == '__main__':
    main()