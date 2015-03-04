import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


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


def process_data(df_raw):
    '''
    Processes raw data - fills missing values and map to numerical format.

    Heuristics:
    loan_status - proceeds reinvested in investment with same risk-return profile
    https://www.lendingclub.com/info/demand-and-credit-profile.action
    
    emp_length - 10 if >10 yrs, average if n/a
    earliest_cr_line - converted to length of time b/w first credit line and 
        date of loan issuance, zero if np.nan
    revol_util - missing values filled in by average
    '''
    df = df_raw[['id', 'grade', 'sub_grade', 'issue_d', 'loan_status', 'int_rate',
                 'loan_amnt', 'term', 
                 'emp_length', 'annual_inc', 'dti',
                 'fico_range_low', 'fico_range_high', 'earliest_cr_line',
                 'open_acc', 'total_acc', 'revol_bal', 'revol_util', 'inq_last_6mths', 
                 'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
                 'purpose', 'home_ownership']]

    df['loan_status'] = df['loan_status'].map({'Fully Paid':1., 'Current':1., 
                                           'In Grace Period':0.76, 
                                           'Late (16-30 days)':0.49, 
                                           'Late (31-120 days)':0.28, 
                                           'Default':0.08,
                                           'Charged Off':0.})

    df['int_rate'] = df['int_rate'].map(lambda x: float(str(x).strip('%')) / 100)

    df['term'] = df['term'].map(lambda x: int(str(x).strip(' months')))

    df['emp_length'] = df['emp_length'].map(lambda x: '0.5 years' if x == '< 1 year' else x)
    df['emp_length'] = df['emp_length'].map(lambda x: '10 years' if x == '10+ years' else x)
    df['emp_length'] = df['emp_length'].map(lambda x: '-1 years' if x == 'n/a' else x)
    df['emp_length'] = df['emp_length'].map(lambda x: float(x.strip(' years')))
    emp_length_mean = np.mean([x for x in df['emp_length'].values if x > 0])
    df['emp_length'] = df['emp_length'].map(lambda x: emp_length_mean if x < 0 else x)

    df['annual_inc'] = df['annual_inc'].map(lambda x: float(x) / 12)
    df.rename(columns={'annual_inc': 'monthly_inc'}, inplace=True)

    df['fico_range_low'] = (df['fico_range_low'] + df['fico_range_high']) / 2.
    df.rename(columns={'fico_range_low': 'fico'}, inplace=True)

    
    df['earliest_cr_line'] = df[['earliest_cr_line', 'issue_d']].apply(lambda x:\
                                x['issue_d'] if pd.isnull(x['earliest_cr_line']) \
                                             else x['earliest_cr_line'], axis=1)

    df['earliest_cr_line'] = df[['earliest_cr_line', 'issue_d']].apply(lambda x:\
                                (datetime.strptime(x['issue_d'], '%b-%Y') \
                                - datetime.strptime(x['earliest_cr_line'], '%b-%Y')).days / 30, axis=1)

    df['revol_util'] = df['revol_util'].map(lambda x: float(str(x).strip('%')) / 100)
    revol_util_mean = np.mean(df['revol_util'])
    df['revol_util'] = df['revol_util'].fillna(revol_util_mean)

    df.rename(columns={'collections_12_mths_ex_med': 'collect_12mths'}, inplace=True)

    df['purpose'] = df['purpose'].map({'credit_card':'credit', 'debt_consolidation':'debt', 
                                       'home_improvement':'home', 'major_purchase':'buy', 
                                       'medical':'medic', 'renewable_energy':'energy', 
                                       'small_business':'biz', 'vacation':'vac', 
                                       'wedding':'wed'})

    df = pd.concat([df, pd.get_dummies(df['purpose'], prefix='purpose')], axis=1)

    df['home_ownership'] = df['home_ownership'].map(lambda x: x.lower())
    df.rename(columns={'home_ownership': 'home_own'}, inplace=True)
    df = pd.concat([df, pd.get_dummies(df['home_own'], prefix='home_own')], axis=1)

    df = df.drop((['fico_range_high', 'purpose', 'home_own']), axis=1)

    return df


def convert_to_array(data,
                     date_range,
                     features,
                     create_label=False,
                     label='sub_grade',
                     label_one='A1',
                     label_zero='D5'):

    date_mask = data['issue_d'].isin(date_range)

    if create_label:
        label_mask = data[label].isin([label_one, label_zero])
        X = data[date_mask & label_mask][features].values
        y = data[date_mask & label_mask][label].map({label_one:1, label_zero:0}).values 
    else:
        X = data[date_mask][features].values
        y = data[date_mask][label].values 

    return X, y


def main():
    # Load data, then pre-process
    print "Loading data..."

    df_3c = pd.read_csv('../data/LoanStats3c_securev1.csv', header=True).iloc[:-2, :]
    df_3b = pd.read_csv('../data/LoanStats3b_securev1.csv', header=True).iloc[:-2, :]
    df_raw = pd.concat((df_3c, df_3b), axis=0)

    df = process_data(df_raw)
    df = df[df['term'] == 36]

    # pickle.dump(df, open('../pickle/df.pkl', 'w'))
    # df = pickle.load(open('../pickle/df.pkl', 'r'))


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

    X, y = convert_to_array(data=df, 
                            date_range=date_range,
                            features=features,
                            create_label=True,
                            label='sub_grade',
                            label_one='A1',
                            label_zero='D5')

    # pickle.dump(X, open('../pickle/X.pkl', 'w'))
    # pickle.dump(y, open('../pickle/y.pkl', 'w'))
    # X = pickle.load(open('../pickle/X.pkl', 'r'))
    # y = pickle.load(open('../pickle/y.pkl', 'r'))


    # Test model accuracy
    print "Testing model accuracy..."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RankingModel(model=RandomForestClassifier,
                         parameters={'n_estimators':100,
                                     'max_depth':10})

    model.fit(X_train, y_train)
    pickle.dump(y, open('../pickle/rankingmodel.pkl', 'w'))

    y_predict = model.predict(X_test)

    print np.sum(y_predict == y_test) / float(len(y_test))


if __name__ == '__main__':
    main()