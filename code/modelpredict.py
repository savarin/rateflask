import numpy as np
import pandas as pd
from preprocessing import dump_to_pickle, load_from_pickle, process_features, process_payment
from maturedmodel import actual_IRR


def actual_matured():
    # Load data, then pre-process
    print "Loading data..."

    df_3a = pd.read_csv('../data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]
    df_raw = df_3a.copy()
    

    # Pre-process data
    print "Pre-processing data..."

    df = process_payment(df_raw)


    # Calculating actual IRR for loans already matured with Dec 2014 int_rate
    print "Calculating IRR..."

    # Replace int_rate with values set in Dec 2015 by sub_grade
    int_rate_dict = {'A1':0.0603, 'A2':0.0649, 'A3':0.0699, 'A4':0.0749, 'A5':0.0819,
                     'B1':0.0867, 'B2':0.0949, 'B3':0.1049, 'B4':0.1144, 'B5':0.1199,
                     'C1':0.1239, 'C2':0.1299, 'C3':0.1366, 'C4':0.1431, 'C5':0.1499,
                     'D1':0.1559, 'D2':0.1599, 'D3':0.1649, 'D4':0.1714, 'D5':0.1786}
    
    IRR = actual_IRR(df, False, int_rate_dict)
    print IRR[:10]

    dump_to_pickle(IRR, '../pickle/IRR_actual_matured_201412rate_20150308.pkl')


def expected_matured():
    # Load data, then pre-process
    print "Loading data..."

    df_3a = pd.read_csv('../data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]
    df_raw = df_3a.copy()
    

    # Pre-process data
    print "Pre-processing data..."

    df = process_features(df_raw, True, False)


    # Loading models
    print "Loading models..."

    model = load_from_pickle('../pickle/StatusModels_20150308.pkl')


    # Calculating expected IRR for loans already matured
    print "Calculating IRR..."

    int_rate_dict = {'A1':0.0603, 'A2':0.0649, 'A3':0.0699, 'A4':0.0749, 'A5':0.0819,
                     'B1':0.0867, 'B2':0.0949, 'B3':0.1049, 'B4':0.1144, 'B5':0.1199,
                     'C1':0.1239, 'C2':0.1299, 'C3':0.1366, 'C4':0.1431, 'C5':0.1499,
                     'D1':0.1559, 'D2':0.1599, 'D3':0.1649, 'D4':0.1714, 'D5':0.1786}

    IRR = model.expected_IRR(df, False, int_rate_dict)
    print IRR[:10]

    dump_to_pickle(IRR, '../pickle/IRR_expected_matured_201412rate_20150308.pkl')


if __name__ == '__main__':
    actual_matured()
    expected_matured()