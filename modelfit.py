import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from preprocessing import dump_to_pickle, load_from_pickle, process_features, process_payment
from currentmodel import StatusModels
from maturedmodel import actual_IRR


def fit_current():
    # Load data, then pre-process
    print "Loading data..."

    df_3c = pd.read_csv('../data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
    df_3b = pd.read_csv('../data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    df_raw = pd.concat((df_3c, df_3b), axis=0)
    

    # Pre-process data
    print "Pre-processing data..."

    df = process_features(df_raw)


    # Train models for every grade for every month
    print "Training models..."

    model = StatusModels(model=RandomForestRegressor,
                         parameters={'n_estimators':100,
                                     'max_depth':10})

    model.train_status_models(df)
    
    dump_to_pickle(model, '../pickle/StatusModels_20150308.pkl')


    # Testing IRR calculations
    print "Testing IRR calculations..."
  
    int_rate_dict = {'A1':0.0603, 'A2':0.0649, 'A3':0.0699, 'A4':0.0749, 'A5':0.0819,
                     'B1':0.0867, 'B2':0.0949, 'B3':0.1049, 'B4':0.1144, 'B5':0.1199,
                     'C1':0.1239, 'C2':0.1299, 'C3':0.1366, 'C4':0.1431, 'C5':0.1499,
                     'D1':0.1559, 'D2':0.1599, 'D3':0.1649, 'D4':0.1714, 'D5':0.1786}

    IRR = model.expected_IRR(df.iloc[:10, :], False, int_rate_dict)
    print IRR


if __name__ == '__main__':
    fit_current()