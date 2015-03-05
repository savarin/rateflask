import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
from preprocessing import dump_to_pickle, load_from_pickle, process_features, convert_to_array
from cashflow import calc_monthly_payments, get_monthly_payments, get_compound_curve, get_cashflows
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
        self.model_dict = defaultdict(list)


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


    def get_expected_payout(self, X, X_sub_grade):
        '''
        Predicts payout probability for whole date range
        '''
        expected_payout = []

        for i, x in enumerate(X):
            expected_payout_x = []
            # X_sub_grade returns a list of sub_grades, X_sub_grade[i][0]
            # returns the string signifying the grade part of the sub_grade,
            # e.g. if X_sub_grade[0] is 'A1', X_sub_grade[0][0] is 'A'.
            for model in self.model_dict[X_sub_grade[i][0]]:
                expected_payout_x.append(model.predict(x))

            # payout_prob_x gives the predicted probability of receiving payment
            # on specified month           
            expected_payout_x = np.array(expected_payout_x).ravel()
            payout_len = expected_payout_x.shape[0]

            beta = curve_fit(self.exponential_dist, 
                             np.arange(1, payout_len + 1),
                             expected_payout_x)[0][0]

            # payout_smooth_x gives the predicted probability after smoothing by
            # fitting to exponential curve with a negative coefficient
            # http://en.wikipedia.org/wiki/Exponential_distribution
            smooth_payout_x = self.exponential_dist(np.arange(1, payout_len + 1),
                                                    beta)

            expected_payout.append(smooth_payout_x)

        return np.array(expected_payout)


    def get_expected_cashflows(self, X, X_sub_grade, X_int_rate, date_range_length):
        '''
        Generates expected cashflow for each loan, i.e. monthly payments 
        multiplied by probability of receiving that payment and compounded to
        the maturity of the loan
        '''
        payout_expected = self.get_expected_payout(X, X_sub_grade)        
        return get_cashflows(payout_expected, X_int_rate, date_range_length)


def main_current():
    # Load data, then pre-process
    print "Loading data..."

    df_3c = pd.read_csv('../data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
    df_3b = pd.read_csv('../data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    df_raw = pd.concat((df_3c, df_3b), axis=0)
    
    df = process_features(df_raw)
    df = df[df['term'] == 36]

    # dump_to_pickle(df, '../pickle/df_prediction.pkl')
    # df = load_from_pickle('../pickle/df_prediction.pkl')


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


    # Train models for every grade for every month
    print "Training models..."

    model = StatusModels(model=RandomForestRegressor,
                         parameters={'n_estimators':100,
                                     'max_depth':10})

    model.train_status_models(df, sub_grade_range, date_range, features)
    
    # dump_to_pickle(model, '../pickle/predictionmodel.pkl')
    # model = load_from_pickle('../pickle/predictionmodel.pkl')


    # Testing cashflow projection
    print "Testing cashflow projection..."

    df_select = df[(df['sub_grade'].isin(sub_grade_range) 
                & (df['issue_d'].isin(date_range)))]

    X = df_select[features].values
    X_sub_grade = df_select['sub_grade'].values
    X_int_rate = df_select['int_rate'].values
    X_id = df_select['id'].values

    cashflows = model.get_expected_cashflows(X, X_sub_grade, X_int_rate, len(date_range))
    IRR = [(np.sum(item))**(1/3.) - 1 for item in cashflows]
    
    # dump_to_pickle(IRR, '../pickle/IRRcurrent.pkl')


def main_matured():
    # Load data, then pre-process
    print "Loading data..."

    df_3a = pd.read_csv('../data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]
    
    df = process_features(df_3a)
    df = df[df['term'] == 36]

    # df = df[df['term'].str.contains('36', na=False)]
    
    # dump_to_pickle(df, '../pickle/df_prediction.pkl')
    # df = load_from_pickle('../pickle/df_prediction.pkl')


    # Define scope
    print "Setting scope..."

    sub_grade_range = ['A1', 'A2', 'A3', 'A4', 'A5', 
                       'B1', 'B2', 'B3', 'B4', 'B5',
                       'C1', 'C2', 'C3', 'C4', 'C5',
                       'D1', 'D2', 'D3', 'D4', 'D5']

    date_range = ['Dec-2011', 'Nov-2011', 'Oct-2011',
                  'Sep-2011', 'Aug-2011', 'Jul-2011', 
                  'Jun-2011', 'May-2011', 'Apr-2011', 
                  'Mar-2011', 'Feb-2011', 'Jan-2011',
                  'Dec-2010', 'Nov-2010', 'Oct-2010', 
                  'Sep-2010', 'Aug-2010', 'Jul-2010', 
                  'Jun-2010', 'May-2010', 'Apr-2010', 
                  'Mar-2010', 'Feb-2010', 'Jan-2010',
                  'Dec-2009', 'Nov-2009', 'Oct-2009', 
                  'Sep-2009', 'Aug-2009', 'Jul-2009', 
                  'Jun-2009', 'May-2009', 'Apr-2009', 
                  'Mar-2009', 'Feb-2009', 'Jan-2009']

    features = ['loan_amnt', 'emp_length', 'monthly_inc', 'dti',
                'fico', 'earliest_cr_line', 'open_acc', 'total_acc', 
                'revol_bal', 'revol_util', 'inq_last_6mths', 
                'delinq_2yrs', 'pub_rec', 'collect_12mths', 
                'purpose_biz', 'purpose_buy', 'purpose_credit', 
                'purpose_debt', 'purpose_energy', 'purpose_home', 
                'purpose_medic', 'purpose_vac', 'purpose_wed', 
                'home_own_any', 'home_own_mortgage', 'home_own_none', 
                'home_own_other', 'home_own_own', 'home_own_rent']

    int_rate_dict = {'A1':0.060299999999999999,
                     'A2':0.064899999999999999,
                     'A3':0.069900000000000004,
                     'A4':0.074900000000000008,
                     'A5':0.081900000000000001,
                     'B1':0.086699999999999999,
                     'B2':0.094899999999999998,
                     'B3':0.10490000000000001,
                     'B4':0.1144,
                     'B5':0.11990000000000001,
                     'C1':0.12390000000000001,
                     'C2':0.12990000000000002,
                     'C3':0.1366,
                     'C4':0.1431,
                     'C5':0.14990000000000001,
                     'D1':0.15590000000000001,
                     'D2':0.15990000000000001,
                     'D3':0.16489999999999999,
                     'D4':0.1714,
                     'D5':0.17859999999999998}


    # Train models for every grade for every month
    print "Training models..."

    model = StatusModels(model=RandomForestRegressor,
                         parameters={'n_estimators':100,
                                     'max_depth':10})

    # model.train_status_models(df, sub_grade_range, date_range, features)
    
    # dump_to_pickle(model, '../pickle/predictionmodel.pkl')
    model = load_from_pickle('../pickle/predictionmodel.pkl')


    # Generating cashflows and calculating IRR
    print "Generating cashflows..."

    # Category 'home_own_any' not in features, added for consistency
    df['home_own_any'] = 0

    df_select = df[(df['sub_grade'].isin(sub_grade_range) 
                & (df['issue_d'].isin(date_range)))]

    X = df_select[features].values
    X_sub_grade = df_select['sub_grade'].values
    X_int_rate = df_select['sub_grade'].map(int_rate_dict).values
    X_id = df_select['id'].values

    cashflows = model.get_expected_cashflows(X, X_sub_grade, X_int_rate, len(date_range))
    
    print "Calculating IRR"
    IRR = [(np.sum(item))**(1/3.) - 1 for item in cashflows]
    dump_to_pickle(IRR, '../pickle/IRR_matured_predicted.pkl')


if __name__ == '__main__':
    # main_current()
    main_matured()