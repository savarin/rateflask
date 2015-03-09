
import numpy as np
import pandas as pd
from lendingclub import LendingClub
from lendingclub.filters import Filter
from pymongo import MongoClient, Connection
from preprocessing import dump_to_pickle, load_from_pickle, process_features
from datetime import datetime
import psycopg2
import time
import re


def request_loan_data(filter_dict):
    '''
    Requests list of loans that can be invested in, then makes individual call
    for details of the loans. Results stored in MongoDB database.
    '''
    # print "Initializing MongoDB database..."
    # client = MongoClient()
    # db = client.lending_test   
    # collection_search = db.collection_search
    # collection_get = db.collection_get

    print "Initializing APIs..."
    club = LendingClub()
    filter_search = Filter(filter_dict)
    club.authenticate()

    print "Requesting loans..." 
    loan_results = club.search(filter_search, start_index=0, limit=1000)
    loans = loan_results['loans']
    loan_ids = [loan['loan_id'] for loan in loans]

    loan_details = []
    for loan_id in loan_ids:
        print "loan_id", loan_id
        request = club.session.get('/browse/loanDetailAj.action', query={'loan_id': loan_id})
        loan_details.append(request.json())
        time.sleep(1)

    # print "Inserting data to database"
    # collection_search.insert(loan_results)
    # collection_get.insert(loan_details)

    return loan_results, loan_details


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


def results_to_database(database_name, table_name, results):
    '''
    Insert loan features and expected IRR into PostgresQL database.
    '''
    conn = psycopg2.connect(dbname=database_name, user='postgres', host='/tmp')
    c = conn.cursor()

    c.execute(
        '''
        CREATE TABLE IF NOT EXISTS {}
            (
                id VARCHAR (50) PRIMARY KEY,
                name VARCHAR (50),
                number VARCHAR (50)
            )
        '''.format(table_name)
    )

    for result in results:
        c.execute(
            ''' 
            INSERT INTO trends
            VALUES ({}, {}, {});
            '''.format(*result)

        )

    conn.commit()
    conn.close()


def main():
    filter_search = {'exclude_existing': False,
                     'funding_progress': 0,
                     'grades': {'All': False,
                                'A': True,
                                'B': True,
                                'C': True,
                                'D': True,
                                'E': False,
                                'F': False,
                                'G': False},
                     'term': {'Year3': True, 'Year5': False}}

    loan_results, loan_details = request_loan_data(filter_search)

    # dump_to_pickle(loan_results, '../pickle/loan_search_20150306_1111.pkl')
    # dump_to_pickle(loan_details, '../pickle/loan_get_20150306_1111.pkl')
    # loan_results = load_from_pickle('../pickle/loan_search_ABCD.pkl')
    # loan_details = load_from_pickle('../pickle/loan_get_ABCD.pkl')

    df_raw = process_requests(loan_results, loan_details)
    print df_raw.shape

    df = process_features(df_raw, False)
    print df.shape


if __name__ == '__main__':
    main()