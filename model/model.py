import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
from transfers.fileio import dump_to_pickle, load_from_pickle
from helpers.preprocessing import process_features
from helpers.cashflow import get_cashflows, calc_IRR


class StatusModel(object):
    def __init__(self,
                 model=RandomForestRegressor,
                 parameters={'n_estimators':100,
                             'max_depth':10}):
        '''
        Model to calculate expected IRR based on loan features, e.g. FICO score.
        Training data is set of 3-year loans issued between 2012 and 2014, i.e.
        not yet matured. Composed of 4x36 models, i.e. one for each grade (A, B,
        C, and D) - month pair (Dec 2014 - Jan 2012).

        Parameters:
        model: Scikit-learn regression model. Random Forest Regressor chosen as
        default due to excellent 'out-of-the-box' performance and versatility.
        sklearn class.
        parameters: Parameters for sklearn class. dictionary.
        '''
        self.model = model
        self.parameters = parameters
        self.model_dict = defaultdict(list)
        self.features_dict = {}
        self.term = 3
        self.grade_range = ['A', 'B', 'C', 'D']

        self.date_range = ['Dec-2014', 'Nov-2014', 'Oct-2014',
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


    def train_model(self, df):
        '''
        Trains model for every grade for whole date range, generating 4x36
        models for every grade-month pair.

        Parameters:
        df: Training data with 36 features. pandas dataframe.
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

        emp_length_mean = np.mean([x for x in df['emp_length'].values if x > 0])
        revol_util_mean = np.mean(df['revol_util'])

        self.features_dict = {'emp_length': emp_length_mean,
                             'revol_util': revol_util_mean}

    def exponential_dist(self, x, beta):
        '''
        Exponential curve for payout probability smoothing.

        Parameters:
        x: Value of x. float.
        beta: Beta coefficient, to be fitted in smoothing process. float.
        '''
        return np.exp(-x / beta)


    def get_expected_payout(self, X, X_sub_grade):
        '''
        Predicts payout probability for whole date range, i.e. calculates
        likelihood of receiving a particular payment on a particular month, for
        36 months.

        Parameters:
        X: Values pertaining to loan features, with n columns. numpy array.
        X_sub_grade: Sub-grade of each loan in X. numpy array.

        Returns:
        Likelihood of receiving payment on specific date. numpy array.
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
            # fitting to exponential curve with a negative coefficient.
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
        the maturity of the loan.

        Parameters:
        X: Values pertaining to loan features, with n columns. numpy array.
        X_int_rate: Interest rate paid by each loan in X, 1-dimensional. numpy array.
        X_compound_rate: Compounding rate for time value of money calculations, 
        1-dimensional. numpy array.
        X_sub_grade: Sub-grade of each loan in X, 1-dimensional. numpy array.
        date_range_length: Total length of calculation period, normally 36. integer.

        Returns:
        Expected cashflow over the life of the loan. numpy array. 
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
        Calculates expected IRR, i.e. the cube root of the sum of all the
        cashflows post-adjustment by risk and time. Expected IRR figure allows
        comparisons to be made between loans of the same sub-grade.

        Parameters:
        df: Training data with n features. pandas dataframe.
        actual_rate: Choice as to whether actual interest rate is to be used, or
        a custom entry. Custom interest rates are generally chosen to allow for
        comparison of loans of the same sub-grade over time. boolean
        rate_dict: Custom interest rate dictionary if actual_rate was True. Key-
        value pair of loan sub-grade and interest rate. dictionary.
        actual_as_compound: Choice as to whether actual interest rate is to be
        used for time value of money calculations. boolean.
        compound_rate: Custom interest rate if actual_as_compound was true. float.

        Returns:
        Expected IRR of each loan. list of floats.
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