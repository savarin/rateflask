import numpy as np
import pandas as pd
from retrieve import request_loan_data, process_requests, results_to_database
from preprocessing import dump_to_pickle, load_from_pickle, process_features
from currentmodel import StatusModels
from flask import Flask, render_template

app = Flask(__name__)

data = []

for i in xrange(10):
    row = []
    for j in xrange(6): 
        row.append(i * j)
    data.append(row)



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
def expected_return():
    print "Loading data..."
    # loan_results, loan_details = request_loan_data(filter_search)
    loan_results = load_from_pickle('pickle/loan_search_ABCD.pkl')
    loan_details = load_from_pickle('pickle/loan_get_ABCD.pkl')

    print "Pre-processing data..."
    df_raw = process_requests(loan_results, loan_details)
    df = process_features(df_raw, False)

    print "Loading models..."
    model = load_from_pickle('../pickle/StatusModels_20150309.pkl')
    IRR = model.expected_IRR(df, True)


    # print "Inserting data to database"
    # df_results = df[['id', 'sub_grade']].copy()
    # df_results['IRR'] = IRR
    # df_results['sub_grade'] = df_results['sub_grade'].map(lambda x: "\'" + str(x) + "\'")
    # results = df_results.values

    # database_name = 'rateflasktest'
    # table_name = 'results'
    # results_to_database(database_name, table_name, results)

    print "Calculating results for display"
    df_retrieve = pd.DataFrame(loan_results)
    percent_fund = df_retrieve[['loanAmountRequested', 'loanAmtRemaining']]\
                                .apply(lambda x: 1 - x['loanAmtRemaining'] \
                                        / float(x['loanAmountRequested']), axis=1).values

    df_display = df[['id', 'sub_grade', 'loan_amnt', 'term', 'int_rate']].copy()
    df_display['percent_fund'] = percent_fund
    df_display['IRR'] = IRR
    df_display['percent_diff'] = df_display[['int_rate', 'IRR']]\
                                    .apply(lambda x: (x['int_rate'] - x['IRR']) \
                                                        / x['int_rate'], axis=1)

    df_display['term'] = df_display['term'].map(lambda x: str(x) + ' mth')
    df_display['loan_amnt'] = df_display['loan_amnt'].map(lambda x: str(x/1000) + ',' + str(x)[-3:])
    df_display['percent_fund'] = df_display['percent_fund'].map(lambda x: str(round(x*100)))
    df_display['int_rate'] = df_display['int_rate'].map(lambda x: str(round(x*100,2)) + '%')
    df_display['IRR'] = df_display['IRR'].map(lambda x: str(round(x*100,2)) + '%')
    df_display['percent_diff'] = df_display['percent_diff'].map(lambda x: str(round(x*100)))

    df_display = df_display[['id', 'sub_grade', 'term', 'loan_amnt', 
                             'percent_fund', 'int_rate', 'IRR', 'percent_diff']]
    data = df_display.values

    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)