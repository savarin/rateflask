import numpy as np
import pandas as pd
from retrieve import request_loan_data, process_requests
from preprocessing import dump_to_pickle, load_from_pickle, process_features
from predictionmodel import StatusModels
from flask import Flask
from nvd3 import discreteBarChart
app = Flask(__name__)

loan_results = load_from_pickle('../pickle/loan_search_ABCD.pkl')
loan_details = load_from_pickle('../pickle/loan_get_ABCD.pkl')

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

features = ['loan_amnt', 'emp_length', 'monthly_inc', 'dti',
            'fico', 'earliest_cr_line', 'open_acc', 'total_acc', 
            'revol_bal', 'revol_util', 'inq_last_6mths', 
            'delinq_2yrs', 'pub_rec', 'collect_12mths', 
            'purpose_biz', 'purpose_buy', 'purpose_credit', 
            'purpose_debt', 'purpose_energy', 'purpose_home', 
            'purpose_medic', 'purpose_vac', 'purpose_wed', 
            'home_own_any', 'home_own_mortgage', 'home_own_none', 
            'home_own_other', 'home_own_own', 'home_own_rent']


@app.route('/')
def index():
    return '<h1> Hello World! </h1>'


@app.route('/expected_return')
def expected_return():
    loan_results, loan_details = request_loan_data(filter_search)

    df_raw = process_requests(loan_results, loan_details)
    df = process_features(df_raw, False)

    df['purpose_energy'] = 0
    df['purpose_wed'] = 0
    df['home_own_any'] = 0
    df['home_own_none'] = 0
    df['home_own_other'] = 0

    model = load_from_pickle('../pickle/predictionmodel.pkl')
    IRR = model.expected_IRR(df, features, True)

    # dump_to_pickle(IRR, '../pickle/IRR_test')
    # IRR = load_from_pickle('../pickle/IRR_test')

    chart_type = 'discreteBarChart'
    chart = discreteBarChart(name=chart_type, color_category='category20c', height=772, width=1250)
    
    xdata = range(len(IRR))[:20]
    ydata = IRR[:20]
    extra_serie = {"tooltip": {"y_start": "", "y_end": ""}}
    chart.add_serie(y=ydata, x=xdata, extra=extra_serie)
    chart.buildcontent()
    body = chart.htmlcontent

    doctype = '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="static/d3.js"></script>
            <script src="static/nv.d3.js"></script>
            <link href="static/nv.d3.css" rel="stylesheet" type="text/css">

            <style>
                text {
                    font: 12px sans-serif;
                }
                svg {
                    display: block;
                }
                html, body, #chart1, svg {
                    margin: 0px;
                    padding: 0px;
                    height: 100%;
                    width: 100%;
                }
            </style>
        </head>
        <body>

    '''

    return doctype + body



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)