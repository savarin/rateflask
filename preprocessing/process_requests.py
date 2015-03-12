import numpy as np
import pandas as pd
from datetime import datetime
import re


def process_requests(loan_results, loan_details):
    '''
    Process data received from API requests, to enable result to be passed
    through process_features
    '''
    df_search = pd.DataFrame(loan_results)\
                    [['loan_id', 'loanGrade', 'loanRate', 
                      'loanAmountRequested', 'loanLength', 'fico', 'purpose']]

    df_get = pd.DataFrame(loan_details)\
                [['completeTenure', 'grossIncome', 'DTI', 
                  'earliestCreditLine', 'openCreditLines', 'totalCreditLines', 
                  'revolvingCreditBalance', 'revolvingLineUtilization', 
                  'inquiriesLast6Months', 'lateLast2yrs', 'publicRecordsOnFile',
                  'monthsSinceLastDelinquency', 'monthsSinceLastRecord',
                  'monthsSinceLastMajorDerogatory',
                  'collectionsExcludingMedical', 'homeOwnership']]

    df_raw = pd.concat((df_search, df_get), axis=1)

    # Introduce new features, not in API request
    df_raw['grade'] = df_raw['loanGrade'].map(lambda x: x[0])
    df_raw['issue_d'] = datetime.now().strftime('%b-%Y')
    df_raw['loan_status'] = 'Current'
    df_raw['fico_range_high'] = df_raw['fico'].map(lambda x: int(x.split('-')[1]))

    # Reformat existing features
    df_raw['loanLength'] = df_raw['loanLength'].map(lambda x: str(x))
    df_raw['grossIncome'] = df_raw['grossIncome']\
                .map(lambda x: int(re.sub("[^0-9]", "", x)) * 12)
    df_raw['fico'] = df_raw['fico'].map(lambda x: int(x.split('-')[0]))
    df_raw['earliestCreditLine'] = df_raw['earliestCreditLine']\
                .map(lambda x: datetime.strptime(x, '%m/%Y').strftime('%b-%Y'))
    df_raw['revolvingCreditBalance'] = df_raw['revolvingCreditBalance']\
                .map(lambda x: int(re.sub("[^0-9]", "", x)) / 100)

    # Reorder and rename columns
    df_raw = df_raw[['loan_id', 'grade', 'loanGrade', 'issue_d', 'loan_status', 'loanRate',
             'loanAmountRequested', 'loanLength', 
             'completeTenure', 'grossIncome', 'DTI',
             'fico', 'fico_range_high', 'earliestCreditLine',
             'openCreditLines', 'totalCreditLines', 'revolvingCreditBalance', 
             'revolvingLineUtilization', 'inquiriesLast6Months',
             'lateLast2yrs', 'publicRecordsOnFile', 'collectionsExcludingMedical',
             'monthsSinceLastDelinquency', 'monthsSinceLastRecord',
             'monthsSinceLastMajorDerogatory',
             'purpose', 'homeOwnership']]

    df_raw.columns = ['id', 'grade', 'sub_grade', 'issue_d', 'loan_status', 'int_rate',
                  'loan_amnt', 'term', 
                  'emp_length', 'annual_inc', 'dti',
                  'fico_range_low', 'fico_range_high', 'earliest_cr_line',
                  'open_acc', 'total_acc', 'revol_bal', 'revol_util', 'inq_last_6mths', 
                  'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
                  'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog',
                  'purpose', 'home_ownership']

    return df_raw