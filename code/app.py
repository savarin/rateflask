import numpy as np
import pandas as pd
from retrieve import request_loan_data, process_requests, results_to_database
from preprocessing import dump_to_pickle, load_from_pickle, process_features
from currentmodel import StatusModels
from flask import Flask, render_template

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
def expected_return():
    # loan_results, loan_details = request_loan_data(filter_search, lending_test)
    # loan_results = load_from_pickle('../pickle/loan_search_ABCD.pkl')
    # loan_details = load_from_pickle('../pickle/loan_get_ABCD.pkl')

    # df_raw = process_requests(loan_results, loan_details)
    # df = process_features(df_raw, False)

    # model = load_from_pickle('../pickle/StatusModels_20150309.pkl')
    # IRR = model.expected_IRR(df, True)

    # df_results = df['id', 'sub_grade']
    # df_results['IRR'] = IRR
    # df_results['sub_grade'] = df_results['sub_grade'].map(lambda x: "\'" + str(x) + "\'")
    # results = df_results.values

    # database_name = 'testinput'
    # table_name = 'results'
    # results_to_database(database_name, table_name, results)

    return render_template('chart_app.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)