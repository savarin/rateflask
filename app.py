import numpy as np
import pandas as pd
from flask import Flask, render_template
from fileio import dump_to_pickle, load_from_pickle
from retrieve import retrieve_loan_data
# from database import insert_into_mongodb, insert_into_postgresql
from preprocessing import process_requests, process_features
from models import currentmodel


app = Flask(__name__)

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


@app.route('/')
def rateflask():
    print "Requesting loan details..."
    loan_results, loan_details = retrieve_loan_data(filter_search)
    # loan_results = load_from_pickle('../pickle/loan_search_ABCD.pkl')
    # loan_details = load_from_pickle('../pickle/loan_get_ABCD.pkl')

    print "Inserting results of API request to database..."
    # insert_into_mongodb(loan_results, loan_details)

    print "Pre-processing data..."
    df_raw = process_requests(loan_results, loan_details)
    df = process_features(df_raw, False)

    print "Loading models..."
    model = load_from_pickle('pickle/model_test.pkl')
    
    print "Calculating results for display..."
    IRR = model.expected_IRR(df, True)
    percent_fund = pd.DataFrame(loan_results)[['loanAmountRequested', 'loanAmtRemaining']]\
                                .apply(lambda x: 1 - x['loanAmtRemaining'] \
                                        / float(x['loanAmountRequested']), axis=1).values

    df_display = df[['id', 'sub_grade', 'term', 'loan_amnt', 'int_rate']].copy()
    df_display['percent_fund'] = percent_fund
    df_display['IRR'] = IRR
    df_display['percent_diff'] = df_display[['int_rate', 'IRR']]\
                                    .apply(lambda x: (x['int_rate'] - x['IRR']) \
                                                        / x['int_rate'], axis=1)

    df_display = df_display[['id', 'sub_grade', 'term', 'loan_amnt', 
                             'percent_fund', 'int_rate', 'IRR', 'percent_diff']]

    print "Inserting processed data to database..."
    # df_results = df_display.copy()
    # df_results['sub_grade'] = df_results['sub_grade'].map(lambda x: "\'" + str(x) + "\'")
    # results = df_results.values

    # database_name = 'rateflask'
    # table_name = 'results'
    # insert_into_postgresql(database_name, table_name, results)

    print "Reformatting for display..."
    df_display['term'] = df_display['term'].map(lambda x: str(x) + ' mth')
    df_display['loan_amnt'] = df_display['loan_amnt'].map(lambda x: '$' \
                                                        + str(x/1000) + ',' + str(x)[-3:])
    df_display['percent_fund'] = df_display['percent_fund'].map(lambda x: str(round(x*100,0)))
    df_display['int_rate'] = df_display['int_rate'].map(lambda x: str(round(x*100,2)) + '%')
    df_display['IRR'] = df_display['IRR'].map(lambda x: str(round(x*100,2)) + '%')
    df_display['percent_diff'] = df_display['percent_diff'].map(lambda x: str(round(x*100,0)))
    
    data = df_display.values
    
    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)