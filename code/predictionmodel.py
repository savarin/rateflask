import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
from rankingmodel import process_data, convert_to_array
import pickle


class StatusModels(object):
    def __init__(self,
                 model=RandomForestRegressor(),
                 parameters={'n_estimators':100,
                             'max_depth':10}):
        '''
        model: choice of model
        parameters: parameter for model, dictionary
        '''
        self.model = model
        self.parameters = parameters
        self.model_dict = {}


    def train_status_models(self, df, sub_grade_range, date_range, features):
        '''
        Trains model for every grade for whole date_range
        '''
        grade_dict = defaultdict(list)        
        grade_range = sorted(list(set([sub_grade[0] for sub_grade in sub_grade_range])))
        
        for grade in grade_range:
            for month in date_range:
                df_select = df[(df['grade'].isin([grade])) 
                             & (df['issue_d'].isin([month]))]
                
                X = df_select[features].values
                y = df_select['loan_status'].values

                model = self.model(**self.parameters)
                model.fit(X, y)

                grade_dict[grade].append(model)
            print grade, 'training completed...'

        self.model_dict = grade_dict


def main():
    # Load data, then pre-process
    print "Loading data..."

    df_3c = pd.read_csv('../data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
    df_3b = pd.read_csv('../data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    df_raw = pd.concat((df_3c, df_3b), axis=0)

    df = process_data(df_raw)
    df = df[df['term'] == 36]

    # pickle.dump(df, open('../pickle/df_prediction.pkl', 'w'))
    # df = pickle.load(open('../pickle/df_prediction.pkl', 'r'))


    # Define scope
    print "Setting scope..."

    sub_grade_range = ['A1', 'A2', 'A3', 'A4', 'A5', 
                       'B1', 'B2', 'B3', 'B4', 'B5',
                       'C1', 'C2', 'C3', 'C4', 'C5',
                       'D1', 'D2', 'D3', 'D4', 'D5']

    date_range = ['Dec-2014', 'Nov-2014', 'Oct-2014',
                  'Sep-2014', 'Aug-2014', 'Jul-2014', 
                  'Jun-2014', 'May-2014', 'Apr-2014', 
                  'Mar-2014', 'Feb-2014', 'Jan-2014',
                  'Dec-2013', 'Nov-2013', 'Oct-2013', 
                  'Sep-2013', 'Aug-2013', 'Jul-2013', 
                  'Jun-2013', 'May-2013', 'Apr-2013', 
                  'Mar-2013', 'Feb-2013', 'Jan-2013',
                  'Dec-2012', 'Nov-2012', 'Oct-2012', 
                  'Sep-2012', 'Aug-2012', 'Jul-2012', 
                  'Jun-2012', 'May-2012', 'Apr-2012', 
                  'Mar-2012', 'Feb-2012', 'Jan-2012']

    features = ['loan_amnt', 'emp_length', 'monthly_inc', 'dti',
                'fico', 'earliest_cr_line', 'open_acc', 'total_acc', 
                'revol_bal', 'revol_util', 'inq_last_6mths', 
                'delinq_2yrs', 'pub_rec', 'collect_12mths', 
                'purpose_biz', 'purpose_buy', 'purpose_credit', 
                'purpose_debt', 'purpose_energy', 'purpose_home', 
                'purpose_medic', 'purpose_vac', 'purpose_wed', 
                'home_own_any', 'home_own_mortgage', 'home_own_none', 
                'home_own_other', 'home_own_own', 'home_own_rent']


    # Train models for every sub-grade for every month
    print "Training models..."

    model = StatusModels(model=RandomForestRegressor,
                         parameters={'n_estimators':100,
                                     'max_depth':10})

    model.train_status_models(df, sub_grade_range, date_range, features)
    
    # pickle.dump(model.model_dict, open('../pickle/model_dict.pkl', 'w'))
    # model.model_dict = pickle.load(open('../pickle/model_dict.pkl', 'r'))

    
if __name__ == '__main__':
    main()