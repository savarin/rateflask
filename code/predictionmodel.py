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


    def exponential_dist(self, x, beta):
        '''
        Exponential curve for payout probability smoothing
        '''
        return np.exp(-x / beta)


    def predict_payout_prob(self, X, X_sub_grade):
        '''
        Predicts payout probability for whole date range
        '''
        payout_prob = []

        for i, x in enumerate(X):
            payout_prob_x = []
            for model in self.model_dict[X_sub_grade[i][0]]:
                payout_prob_x.append(model.predict(x))

            # payout_prob_x gives the predicted probability of receiving payment
            # on specified month           
            payout_prob_x = np.array(payout_prob_x).ravel()
            payout_len = payout_prob_x.shape[0]

            beta = curve_fit(self.exponential_dist, 
                             np.arange(1, payout_len + 1),
                             payout_prob_x)[0][0]

            # payout_smooth_x gives the predicted probability after smoothing by
            # fitting to exponential curve with a negative coefficient
            # http://en.wikipedia.org/wiki/Exponential_distribution
            payout_smooth_x = self.exponential_dist(np.arange(1, payout_len + 1),
                                                    beta)

            payout_prob.append(payout_smooth_x)

        return np.array(payout_prob)


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


    # Testing cashflow projection
    print "Testing cashflow projection..."

    df_select = df[(df['sub_grade'].isin(sub_grade_range) 
                & (df['issue_d'].isin(date_range)))]

    X = df_select[features].values
    X_sub_grade = df_select['sub_grade'].values
    X_int_rate = df_select['int_rate'].values
    X_id = df_select['id'].values

    print 'payout_prob', model.predict_payout_prob(X, X_sub_grade) 


if __name__ == '__main__':
    main()