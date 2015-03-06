import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import dump_to_pickle, load_from_pickle, process_features, convert_to_array


class RankingModel(object):
    def __init__(self,
                 model=RandomForestClassifier(),
                 parameters={'n_estimators':100,
                             'criterion':'entropy',
                             'max_depth':10}):      
        '''
        model: choice of model
        parameters: parameter for model, dictionary
        '''
        self.model = model
        self.parameters = parameters
        

    def fit(self, X, y):
        '''
        X: 2-dim array representing feature matrix for training data
        y: 1-d array representing labels for training data
        '''
        self.model = self.model(**self.parameters).fit(X, y)


    def predict(self, X):
        '''
        X: 2-dim array representing feature matrix for test data
        '''
        return self.model.predict(X)


def main():
    # Load data, then pre-process
    print "Loading data..."

    df_3c = pd.read_csv('../data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
    df_3b = pd.read_csv('../data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    df_raw = pd.concat((df_3c, df_3b), axis=0)

    df = process_features(df_raw)

    # dump_to_pickle(df, '../pickl/df.pkl')
    # df = load_from_pickle('../pickle/df.pkl')


    # Define scope, then convert data to array
    print "Setting scope..."

    date_range = ['Dec-2014', 'Nov-2014', 'Oct-2014', 
                  'Sep-2014', 'Aug-2014', 'Jul-2014', 
                  'Jun-2014', 'May-2014', 'Apr-2014', 
                  'Mar-2014', 'Feb-2014', 'Jan-2014']

    features = ['loan_amnt', 'emp_length', 'monthly_inc', 'dti', 
                'fico', 'earliest_cr_line', 'open_acc', 'total_acc', 
                'revol_bal', 'revol_util', 'inq_last_6mths', 
                'delinq_2yrs', 'pub_rec', 'collect_12mths', 
                'purpose_biz', 'purpose_buy', 'purpose_credit', 
                'purpose_debt', 'purpose_energy', 'purpose_home', 
                'purpose_medic', 'purpose_vac', 'purpose_wed', 
                'home_own_any', 'home_own_mortgage', 'home_own_none', 
                'home_own_other', 'home_own_own', 'home_own_rent']

    X, y = convert_to_array(df, date_range, features,
                            create_label=True,
                            label='sub_grade',
                            label_one='A1',
                            label_zero='D5')

    # dump_to_pickle(X, '../pickle/X.pkl')
    # dump_to_pickle(y, '../pickle/y.pkl')
    # X = load_from_pickle('../pickle/X.pkl')
    # y = load_from_pickle('../pickle/y.pkl')


    # Test model accuracy
    print "Testing model accuracy..."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RankingModel(model=RandomForestClassifier,
                         parameters={'n_estimators':100,
                                     'max_depth':10})

    model.fit(X_train, y_train)
    
    # dump_to_pickle(model, '../pickle/rankingmodel.pkl')

    y_predict = model.predict(X_test)
    print np.sum(y_predict == y_test) / float(len(y_test))


if __name__ == '__main__':
    main()