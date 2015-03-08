import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
import pickle
from preprocessing import dump_to_pickle, load_from_pickle, process_features
from cashflow import calc_monthly_payment, get_monthly_payments, \
                     get_compound_curve, get_cashflows, calc_IRR


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
        self.term = 3
        self.grade_range = ['A', 'B', 'C', 'D']

        self.date_range = ['Dec-2014', 'Nov-2014', 'Oct-2014',
                           'Sep-2014', 'Aug-2014', 'Jul-2014', 
                           'Jun-2014', 'May-2014', 'Apr-2014', 
                           'Mar-2014', 'Feb-2014', 'Jan-2014',
                           'Dec-2013', 'Nov-2013', 'Oct-2013', 
                           'Sep-2013', 'Aug-2013', 'Jul-2013', 
                           'Jun-2013', 'May-2013', 'Apr-2013', 
                           'Mar-2013', 'Feb-2013', 'Jan-2013',                            'Dec-2012', 'Nov-2012', 'Oct-2012', 
                           'Sep-2012', 'Aug-2012', 'Jul-2012', 
                           'Jun-2012', 'May-2012', 'Apr-2012', 
                           'Mar-2012', 'Feb-2012', 'Jan-2012']

        self.features = ['loan_amnt', 'emp_length', 'monthly_inc', 'dti', 
                         'fico', 'earliest_cr_line', 'open_acc', 'total_acc', 
                         'revol_bal', 'revol_util', 'inq_last_6mths', 
                         'delinq_2yrs', 'pub_rec', 'collect_12mths', 
                         'last_delinq', 'last_record', 'last_derog',
                         'purpose_debt', 'purpose_credit', 'purpose_home', 
                         'purpose_other', 'purpose_buy', 'purpose_biz', 
                         'purpose_medic', 'purpose_car', 'purpose_move', 
                         'purpose_vac', 'purpose_house', 'purpose_wed', 'purpose_energy', 
                         'home_mortgage', 'home_rent', 'home_own',
                         'home_other', 'home_none', 'home_any']


    def train_status_models(self, df):
        '''
        Trains model for every grade for whole date_range
        '''
        grade_dict = defaultdict(list)        
        
        for grade in self.grade_range:
            for month in self.date_range:
                df_select = df[(df['grade'].isin([grade])) 
                             & (df['issue_d'].isin([month]))]
                
                X = df_select[self.features].values
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


    def get_expected_cashflows(self, X, X_int_rate, X_compound_rate, 
                                     X_sub_grade, date_range_length):
        '''
        Generates expected cashflow for each loan, i.e. monthly payments 
        multiplied by probability of receiving that payment and compounded to
        the maturity of the loan
        '''
        expected_payout = self.get_expected_payout(X, X_sub_grade)
        return get_cashflows(expected_payout, X_int_rate, X_compound_rate, 
                             date_range_length)


    def expected_IRR(self, df, 
                           actual_rate=True, 
                           rate_dict={},
                           actual_as_compound=True,
                           compound_rate=0.01):
        '''
        Calculates IRR for loans that have not yet matured.

        actual_rate: If using interest rates in data or custom, Boolean
        rate_dict: Custom dictionary with sub-grade as keys as interest rate as
        values, dictionary

        actual_as_compound: If using interest rates in data as compound rate or 
        custom, Boolean
        compound_rate: Custom interest rate for compounding, float
        '''
        X = df[self.features].values
        date_range_length = self.term * 12

        if actual_rate:
            X_int_rate = df['int_rate'].values
        else:
            X_int_rate = df['sub_grade'].map(rate_dict).values

        if actual_as_compound:
            X_compound_rate = X_int_rate
        else:
            X_compound_rate = np.array([compound_rate] * X_int_rate.shape[0])

        X_sub_grade = df['sub_grade'].values

        expected_cashflows = self.get_expected_cashflows(X, 
                                                         X_int_rate,
                                                         X_compound_rate, 
                                                         X_sub_grade,
                                                         date_range_length)

        return calc_IRR(expected_cashflows, self.term)


def main_test():
    # Load data, then pre-process
    print "Loading data..."

    df_3c = pd.read_csv('../data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
    df_3b = pd.read_csv('../data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    df_raw = pd.concat((df_3c, df_3b), axis=0)
    
    df = process_features(df_raw)

    # dump_to_pickle(df, '../pickle/df_test.pkl')
    # df = load_from_pickle('../pickle/df_test.pkl')


    # Train models for every grade for every month
    print "Training models..."

    model = StatusModels(model=RandomForestRegressor,
                         parameters={'n_estimators':100,
                                     'max_depth':10})

    model.train_status_models(df)
    
    dump_to_pickle(model, '../pickle/model_test.pkl')
    model = load_from_pickle('../pickle/model_test.pkl')


    # Testing IRR calculations
    print "Calculating IRR..."

    IRR = model.expected_IRR(df.iloc[:10, :], True, {}, False)
    print IRR


def main_fit():
    # Load data, then pre-process
    print "Loading data..."

    df_3c = pd.read_csv('../data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
    df_3b = pd.read_csv('../data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    df_raw = pd.concat((df_3c, df_3b), axis=0)
    
    df = process_features(df_raw)

    # dump_to_pickle(df, '../pickle/df.pkl')
    # df = load_from_pickle('../pickle/df.pkl')


    # Define scope
    print "Setting scope..."

    int_rate_dict = {'A1':0.0603, 'A2':0.0649, 'A3':0.0699, 'A4':0.0749, 'A5':0.0819,
                     'B1':0.0867, 'B2':0.0949, 'B3':0.1049, 'B4':0.1144, 'B5':0.1199,
                     'C1':0.1239, 'C2':0.1299, 'C3':0.1366, 'C4':0.1431, 'C5':0.1499,
                     'D1':0.1559, 'D2':0.1599, 'D3':0.1649, 'D4':0.1714, 'D5':0.1786}


    # Train models for every grade for every month
    print "Training models..."

    model = StatusModels(model=RandomForestRegressor,
                         parameters={'n_estimators':100,
                                     'max_depth':10})

    model.train_status_models(df)
    
    dump_to_pickle(model, '../pickle/predictionmodel_20150306.pkl')
    # model = load_from_pickle('../pickle/predictionmodel.pkl')


    # Testing IRR calculations
    print "Calculating IRR..."
    
    # IRR = model.expected_IRR(df, True)
    IRR = model.expected_IRR(df, False, int_rate_dict)
    
    dump_to_pickle(IRR, '../pickle/IRR_current_201412rate_20150307.pkl')


def main_predict():
    # Load data, then pre-process
    print "Loading data..."

    df_3a = pd.read_csv('../data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]
    df_raw = df_3a.copy()
    
    df = process_features(df_raw, True, False)


    # Define scope
    print "Setting scope..."

    int_rate_dict = {'A1':0.0603, 'A2':0.0649, 'A3':0.0699, 'A4':0.0749, 'A5':0.0819,
                     'B1':0.0867, 'B2':0.0949, 'B3':0.1049, 'B4':0.1144, 'B5':0.1199,
                     'C1':0.1239, 'C2':0.1299, 'C3':0.1366, 'C4':0.1431, 'C5':0.1499,
                     'D1':0.1559, 'D2':0.1599, 'D3':0.1649, 'D4':0.1714, 'D5':0.1786}

    model = load_from_pickle('../pickle/predictionmodel_20150306.pkl')


    # Calculating expected IRR for loans already matured
    print "Calculating IRR..."

    IRR = model.expected_IRR(df, False, int_rate_dict)
    
    dump_to_pickle(IRR, '../pickle/IRR_matured_201412rate_20150307.pkl')


if __name__ == '__main__':
    main_test()
    # main_fit()
    # main_predict()