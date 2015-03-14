import numpy as np
import pandas as pd
import sys
import dill
from flask import Flask, render_template, make_response
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict, deque
from functools import wraps, update_wrapper
from datetime import datetime 
from transfers.fileio import dump_to_pickle, load_from_pickle
from transfers.retrieve import request_loan_data
from transfers.database import insert_into_mongodb, insert_into_postgresql
from helpers.preprocessing import process_requests, process_features
from helpers.postprocessing import generate_for_charts, reformat_for_display
from model.model import StatusModel
from model.start import initialize_model


app = Flask(__name__)

DATA = deque('0')
datetime_now = datetime.now()
DATETIME_NOW = [datetime_now.strftime("%b"), datetime_now.day, datetime_now.year,
                datetime_now.strftime("%H"), datetime_now.strftime("%M")]


def nocache(view):
    '''
    Disables caching to allow charts to refresh.
    '''
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, \
                                             post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view) 


def run_process():
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


    print "Requesting loan details..."
    loan_results, loan_details = request_loan_data(filter_search)

    print "Inserting results of API request to database..."
    try:
        insert_into_mongodb(loan_results, loan_details)
    except Exception:
        print "MongoDB connection error, proceeding to next step..."

    print "Loading model..."
    try:
        model = load_from_pickle('pickle/model.pkl')
    except (OSError, IOError):
        print "Model not found. Initializing training process, this might take some time..."
        model = initialize_model()

    print "Pre-processing data..."
    df_raw = process_requests(loan_results, loan_details)
    df = process_features(df_raw, restrict_date=False, features_dict=model.features_dict)

    print "Calculating results for display..."
    IRR = model.expected_IRR(df, True)
    percent_fund = pd.DataFrame(loan_results)[['loanAmountRequested', 'loanAmtRemaining']]\
                                .apply(lambda x: 1 - x['loanAmtRemaining'] \
                                        / float(x['loanAmountRequested']), axis=1).values

    df_display = df[['id', 'sub_grade', 'term', 'loan_amnt', 'int_rate']].copy()
    df_display['datetime_now'] = "\'" + str(datetime_now) + "\'"
    df_display['percent_fund'] = percent_fund
    df_display['IRR'] = IRR
    df_display['percent_diff'] = df_display[['int_rate', 'IRR']]\
                                    .apply(lambda x: (x['int_rate'] - x['IRR']) \
                                                        / x['int_rate'], axis=1)

    df_display = df_display[['id', 'datetime_now', 'sub_grade', 'term', 'loan_amnt', 
                             'percent_fund', 'int_rate', 'IRR', 'percent_diff']]


    print "Inserting processed data to database..."
    df_results = df_display.drop(['percent_fund'], axis=1).copy()
    df_results['sub_grade'] = df_results['sub_grade'].map(lambda x: "\'" + str(x) + "\'")
    results = df_results.values

    database_name = 'rateflask'
    table_name = 'results'
    try:
        insert_into_postgresql(database_name, table_name, results)
    except Exception:
        print "PostgreSQL connection error, proceeding to next step..."

    print "Generating data for charts..."
    df_max = df_display.groupby('sub_grade').max()['IRR']
    generate_for_charts(df_max)

    print "Reformatting for display..."
    df_final = reformat_for_display(df_display)

    print "Process completed, displaying results..."
    data = df_final.values
    return data


@app.route('/refresh')
@nocache
def refresh():
    results = run_process()
    DATA.append(results)
    DATA.popleft()
    
    return "Calculations based on latest update completed."


@app.route('/')
@nocache
def rateflask():
    return render_template('index.html', data=DATA[-1], datetime_now=DATETIME_NOW)


if __name__ == '__main__':
    debug_state = True

    if len(sys.argv) > 1:
        if sys.argv[1] == 'production':
            debug_state = False

    app.run(host='0.0.0.0', port=8000, debug=debug_state)