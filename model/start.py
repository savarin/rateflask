import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from transfers.fileio import dump_to_pickle
from helpers.preprocessing import process_features
from model import StatusModel


def initialize_model():
    model = StatusModel(model=RandomForestRegressor,
                        parameters={'n_estimators':100,
                                     'max_depth':10})

    try:
        df_3c = pd.read_csv('data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
        df_3b = pd.read_csv('data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    except (OSError, IOError):
        print "Training data not found. Please install from https://www.lendingclub.com/info/download-data.action"

    df_train = pd.concat((df_3c, df_3b), axis=0)
    df_train = process_features(df_train, restrict_date=True, current_loans=True)

    model.train_model(df_train)
    dump_to_pickle(model, 'pickle/model.pkl')

    return model