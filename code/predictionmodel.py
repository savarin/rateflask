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


    def get_payout_prob(self, X, X_sub_grade):
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


    def calc_monthly_payments(self, loan_amnt, int_rate, term):
        '''
        Calculates monthly payments (principal + interest) for loan with specified
        term and interest rate
        '''
        monthly_rate = int_rate / 12
        date_range_length = term * 12

        numerator = monthly_rate * ((1 + monthly_rate) ** date_range_length)
        denominator = ((1 + monthly_rate) ** date_range_length) - 1

        return loan_amnt * numerator / denominator


    def get_monthly_payments(self, X_int_rate, date_range_length):
        '''
        Generates monthly payments for each loan
        '''
        monthly_payments = np.ones((X_int_rate.shape[0], date_range_length))

        for i, int_rate in enumerate(X_int_rate):
            monthly_payments[i] = (self.calc_monthly_payments(100, int_rate, 3)\
                                    * monthly_payments[i])   

        return monthly_payments


    def get_compound_curve(self, X_int_rate, date_range_length):
        '''
        Generates compounding curve for each loan, assumes coupon reinvested in 
        investment of similar return
        '''
        compound_curve = []
        
        for i, int_rate in enumerate(X_int_rate):
            compound_curve.append(np.array([(1 + int_rate / 12)**(i-1) for i 
                in xrange(date_range_length, 0, -1)]))

        return np.array(compound_curve)


    def get_expected_cashflows(self, X, X_sub_grade, X_int_rate, date_range_length):
        '''
        Generates expected cashflow for each loan, i.e. monthly payments 
        multiplied by probability of receiving that payment and compounded to
        the maturity of the loan
        '''
        payout_prob = self.get_payout_prob(X, X_sub_grade)        
        monthly_payments = self.get_monthly_payments(X_int_rate, date_range_length)
        compound_curve = self.get_compound_curve(X_int_rate, date_range_length)

        expected_cashflows = []

        for i in xrange(len(payout_prob)):
            cashflow = payout_prob[i] * monthly_payments[i] * compound_curve[i]
            expected_cashflows.append(cashflow)

        return np.array(expected_cashflows)


def main():
    # Load data, then pre-process
    print "Loading data..."

    # df_3c = pd.read_csv('../data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
    # df_3b = pd.read_csv('../data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    # df_raw = pd.concat((df_3c, df_3b), axis=0)

    # df = process_data(df_raw)
    # df = df[df['term'] == 36]

    # pickle.dump(df, open('../pickle/df_prediction.pkl', 'w'))
    df = pickle.load(open('../pickle/df_prediction.pkl', 'r'))


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

    # model.train_status_models(df, sub_grade_range, date_range, features)
    
    # pickle.dump(model.model_dict, open('../pickle/model_dict.pkl', 'w'))
    model.model_dict = pickle.load(open('../pickle/model_dict.pkl', 'r'))


    # Testing cashflow projection
    # print "Testing cashflow projection..."

    sub_grade_range = ['D1']#, 'D2', 'D3', 'D4', 'D5']

    df_select = df[(df['sub_grade'].isin(sub_grade_range) 
                & (df['issue_d'].isin(date_range)))]

    X = df_select[features].values[:2, :]
    X_sub_grade = df_select['sub_grade'].values[:2]
    X_int_rate = df_select['int_rate'].values[:2]
    X_id = df_select['id'].values[:2]

    # get_payout_prob(self, X, X_sub_grade):
    # calc_monthly_payments(self, loan_amnt, int_rate, term)
    # get_monthly_payments(self, X_int_rate, date_range_length)
    # get_compound_curve(self, X_int_rate, date_range_length)

    # print 'payout_prob', model.get_payout_prob(X, X_sub_grade) 
    # print 'monthly_payments', model.get_monthly_payments(X_int_rate, len(date_range))
    # print 'compound curve', model.get_compound_curve(X_int_rate, len(date_range))

    cashflows = model.get_expected_cashflows(X, X_sub_grade, X_int_rate, len(date_range))
    # print 'cashflows', cashflows
 
    IRR = np.array([((np.sum(item)) / 100) ** (1/3.) - 1 for item in cashflows])

    print X_id.shape
    print X_sub_grade.shape
    print IRR.shape

    print np.concatenate((X_id[:, np.newaxis], X_sub_grade[:, np.newaxis], IRR[:, np.newaxis]), axis=1)

    # pickle.dump(IRR, open('../pickle/results_D.pkl', 'w'))




if __name__ == '__main__':
    main()