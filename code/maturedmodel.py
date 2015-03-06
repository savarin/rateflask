import numpy as np
import pandas as pd
from preprocessing import dump_to_pickle, load_from_pickle, process_payment
from cashflow import calc_monthly_payments, get_monthly_payments, get_compound_curve, get_cashflows


def get_actual_payout(X, date_range_length):
    '''
    Generates actual payout of each loan. If loan pays in full returns array of
    ones of length date arang, otherwise returns arrays of ones for paid 
    months, zeros for unpaid with residual and recovery values filled in in
    respective months
    '''
    payout_actual = []

    for i, x in enumerate(X):
        if (x[0] == 0) or (x[1] >= date_range_length):
            payout_actual_x = [1] * date_range_length
            payout_actual.append(payout_actual_x)
        else:
            payout_actual_x = [1] * int(x[1])
            payout_actual_x.append(x[2])

            m = len(payout_actual_x)
            if m < date_range_length:
                payout_actual_x += ([0] * (date_range_length - m))
            payout_actual_x[-1] += x[3]
            payout_actual.append((payout_actual_x))

    return np.array(payout_actual)


def get_actual_cashflows(X, X_int_rate, date_range_length):
    '''
    Generates actual cashflow for each loan, i.e. monthly payments multiplied
    by actual payment as fraction of installment and compounded to the maturity
    of the loan
    '''
    payout_actual = get_actual_payout(X, date_range_length)
    return get_cashflows(payout_actual, X_int_rate, date_range_length)


def matured_IRR(df_raw, date_range_length, actual_rate=True, rate_dict={}):
    '''
    Calculates IRR for loans that have already matured.
    '''
    df = process_payment(df_raw)

    X = df[['default_status', 'months_paid', 'residual', 'recovery']].values

    if actual_rate:
        X_int_rate = df_raw['int_rate'].map(lambda x: float(str(x).strip('%')) / 100).values
    else:
        X_int_rate = df_raw['sub_grade'].map(rate_dict).values

    loan_id = df_raw['id'].values
    actual_cashflows = get_actual_cashflows(X, X_int_rate, date_range_length)

    return actual_cashflows


def main_basic():
    df_3a = pd.read_csv('../data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]

    df = df_3a.copy()
    df = df[df['term'].str.contains('36', na=False)]
    df = df[df['grade'].isin(['A', 'B', 'C', 'D'])]
    
    df = df[(df['issue_d'].str.contains('2009')) \
         | (df['issue_d'].str.contains('2010')) \
         | (df['issue_d'].str.contains('2011'))]

    IRR = matured_IRR(df, 36, True)
    print IRR

    # dump_to_pickle(IRR, '../pickle/matured_IRR.pkl')


def main_recent():
    # Replace int_rate with values in Dec 2015
    df_3a = pd.read_csv('../data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]

    df = df_3a.copy()
    df = df[df['term'].str.contains('36', na=False)]
    df = df[df['grade'].isin(['A', 'B', 'C', 'D'])]
    
    df = df[(df['issue_d'].str.contains('2009')) \
         | (df['issue_d'].str.contains('2010')) \
         | (df['issue_d'].str.contains('2011'))]

    int_rate_dict = {'A1':0.0603,
                     'A2':0.0649,
                     'A3':0.0699,
                     'A4':0.0749,
                     'A5':0.0819,
                     'B1':0.0867,
                     'B2':0.0949,
                     'B3':0.1049,
                     'B4':0.1144,
                     'B5':0.1199,
                     'C1':0.1239,
                     'C2':0.1299,
                     'C3':0.1366,
                     'C4':0.1431,
                     'C5':0.1499,
                     'D1':0.1559,
                     'D2':0.1599,
                     'D3':0.1649,
                     'D4':0.1714,
                     'D5':0.1786}

    
    print "Generating cashflows..."
    actual_cashflows = matured_IRR(df, 36, False, int_rate_dict)

    print "Calculating IRR..."
    dump_to_pickle(IRR, '../pickle/IRR_matured_actual.pkl')


if __name__ == '__main__':
    # main_basic()
    main_recent()