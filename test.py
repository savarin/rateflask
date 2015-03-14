import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from transfers.fileio import dump_to_pickle, load_from_pickle
from helpers.preprocessing import process_features, process_payment
from model.model import StatusModel
from model.start import initialize_model_class
from model.validate import actual_IRR


def test_expected_current():
    print "Loading data..."
    try:
        df_3c = pd.read_csv('data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
        df_3b = pd.read_csv('data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    except (OSError, IOError):
        print "Training data not found. Please install from https://www.lendingclub.com/info/download-data.action"

    df_raw = pd.concat((df_3c, df_3b), axis=0)
    
    print "Pre-processing data..."
    df = process_features(df_raw)

    print "Initializing model..."
    model = StatusModel(model=RandomForestRegressor,
                        parameters={'n_estimators':100,
                                     'max_depth':10})

    print "Training model..."
    try:
        model = load_from_pickle('pickle/model.pkl')
    except (OSError, IOError):
        print "Model not found. Training model, this might take some time..."
        model.train_model(df)
        dump_to_pickle(model, 'pickle/model.pkl')

    print "Calculating IRR..."
    int_rate_dict = {'A1':0.0603, 'A2':0.0649, 'A3':0.0699, 'A4':0.0749, 'A5':0.0819,
                     'B1':0.0867, 'B2':0.0949, 'B3':0.1049, 'B4':0.1144, 'B5':0.1199,
                     'C1':0.1239, 'C2':0.1299, 'C3':0.1366, 'C4':0.1431, 'C5':0.1499,
                     'D1':0.1559, 'D2':0.1599, 'D3':0.1649, 'D4':0.1714, 'D5':0.1786}

    IRR = model.expected_IRR(df.iloc[:10, :], 
                             actual_rate=True)
    print "Expected IRR calculated with default int_rate:", IRR
    
    IRR = model.expected_IRR(df.iloc[:10, :],
                             actual_rate=False, 
                             rate_dict=int_rate_dict)
    print "Expected IRR calculated with custom int_rate:", IRR

    IRR = model.expected_IRR(df.iloc[:10, :], 
                             actual_rate=True,
                             actual_as_compound=False, 
                             compound_rate=0.01)
    print "Expected IRR calculated with custom compounding curve:", IRR


def test_actual_matured():
    print "Loading data..."
    try:
        df_3a = pd.read_csv('data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]
        df_raw = df_3a.copy()
    except (OSError, IOError):
        print "Training data not found. Please install from https://www.lendingclub.com/info/download-data.action"

    print "Pre-processing data..."
    df = process_payment(df_raw)

    print "Calculating IRR..."
    int_rate_dict = {'A1':0.0603, 'A2':0.0649, 'A3':0.0699, 'A4':0.0749, 'A5':0.0819,
                     'B1':0.0867, 'B2':0.0949, 'B3':0.1049, 'B4':0.1144, 'B5':0.1199,
                     'C1':0.1239, 'C2':0.1299, 'C3':0.1366, 'C4':0.1431, 'C5':0.1499,
                     'D1':0.1559, 'D2':0.1599, 'D3':0.1649, 'D4':0.1714, 'D5':0.1786}
    
    IRR = actual_IRR(df.iloc[:10, :], 
                     actual_rate=True)
    print "Actual IRR calculated with default int_rate:", IRR

    IRR = actual_IRR(df.iloc[:10, :], 
                     actual_rate=False, 
                     rate_dict=int_rate_dict)
    print "Actual IRR calculated with custom int_rate:",  IRR

    IRR = actual_IRR(df.iloc[:10, :], 
                     actual_rate=True, 
                     actual_as_compound=False,
                     compound_rate=0.01)
    print "Actual IRR calculated with custom compounding curve:", IRR


def compare_IRR():
    print "Loading model..."
    try:
        model = load_from_pickle('pickle/test.pkl')
    except (OSError, IOError):
        print "Model not found. Initializing training process, this might take some time..."
        model = initialize_model()

    print "Calculating expected IRR of loans that have matured..."
    df_3a = pd.read_csv('data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]
    df_expected = process_features(df_3a.copy(), restrict_date=True, current_loans=False)

    # To make a fair comparison, the same set of interest rates are used. The 
    # enclosed set is from Dec 2014.
    int_rate_dict = {'A1':0.0603, 'A2':0.0649, 'A3':0.0699, 'A4':0.0749, 'A5':0.0819,
                     'B1':0.0867, 'B2':0.0949, 'B3':0.1049, 'B4':0.1144, 'B5':0.1199,
                     'C1':0.1239, 'C2':0.1299, 'C3':0.1366, 'C4':0.1431, 'C5':0.1499,
                     'D1':0.1559, 'D2':0.1599, 'D3':0.1649, 'D4':0.1714, 'D5':0.1786}

    IRR_predict = model.expected_IRR(df_expected, actual_rate=False, rate_dict=int_rate_dict)

    print "Calculating actual IRR of loans that have matured..."
    df_actual = process_payment(df_3a.copy())
    IRR_true = actual_IRR(df_actual, actual_rate=False, rate_dict=int_rate_dict)

    print "Collecting results..."
    df_result = df_expected[['id', 'sub_grade']].copy()

    df_result['IRR_predict'] = IRR_predict
    df_result['IRR_true'] = IRR_true
    df_result['IRR_difference'] = df_result['IRR_true'] - df_result['IRR_predict']

    print "Comparison of IRR by subgrade:", df_result.groupby('sub_grade').mean()

    print "Plotting results..."
    plt.figure(figsize = (12, 6))
    x = xrange(20)
    y_true = df_result.groupby('sub_grade').mean()['IRR_true']
    y_predict = df_result.groupby('sub_grade').mean()['IRR_predict']

    plt.plot(x, y_true, label='Actual IRR')
    plt.plot(x, y_predict, label='Predicted IRR')
    plt.legend(loc='best')
    plt.xlabel('Sub-grade, 0:A1 19:D5')
    plt.ylabel('IRR')     
    plt.title("Comparison of predicted vs true IRR")
    plt.show()


if __name__ == '__main__':
    # test_expected_current()
    # test_actual_matured()
    compare_IRR()