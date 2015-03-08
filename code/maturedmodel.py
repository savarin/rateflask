import numpy as np
import pandas as pd
from preprocessing import dump_to_pickle, load_from_pickle, process_payment
from cashflow import calc_monthly_payment, get_monthly_payments, get_compound_curve, get_cashflows, calc_IRR


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


def get_actual_cashflows(X, X_int_rate, X_compound_rate, date_range_length):
    '''
    Generates actual cashflow for each loan, i.e. monthly payments multiplied
    by actual payment as fraction of installment and compounded to the maturity
    of the loan
    '''
    payout_actual = get_actual_payout(X, date_range_length)

    return get_cashflows(payout_actual, X_int_rate, X_compound_rate,
                         date_range_length)


def actual_IRR(df, 
               actual_rate=True, 
               rate_dict={},
               actual_as_compound=True,
               compound_rate=0.01):
    '''
    Calculates IRR for loans that have already matured.

    actual_rate: If using interest rates in data or custom, Boolean
    rate_dict: Custom dictionary with sub-grade as keys as interest rate as
    values, dictionary

    actual_as_compound: If using interest rates in data as compound rate or 
    custom, Boolean
    compound_rate: Custom interest rate for compounding, float
    '''
    term = 3
    date_range_length = term * 12

    X = df[['default_status', 'months_paid', 'residual', 'recovery']].values

    if actual_rate:
        X_int_rate = df['int_rate'].values
    else:
        X_int_rate = df['sub_grade'].map(rate_dict).values

    if actual_as_compound:
        X_compound_rate = X_int_rate
    else:
        X_compound_rate = np.array([compound_rate] * X_int_rate.shape[0])

    actual_cashflows = get_actual_cashflows(X, 
                                            X_int_rate, 
                                            X_compound_rate,
                                            date_range_length)

    return calc_IRR(actual_cashflows, term)


def main_actual():
    # Load data, then pre-process
    print "Loading data..."

    df_3a = pd.read_csv('../data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]
    df_raw = df_3a.copy()

    df = process_payment(df_raw)


    # Calculating actual IRR for loans already matured with actual int_rate    
    print "Calculating IRR..."
    
    IRR = actual_IRR(df, True, {}, False)
    
    print IRR[:10]
    # dump_to_pickle(IRR, '../pickle/matured_IRR.pkl')


def main_recent():
    # Load data, then pre-process
    print "Loading data..."

    df_3a = pd.read_csv('../data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]
    df_raw = df_3a.copy()
    
    df = process_payment(df_raw)


    # Replace int_rate with values set in Dec 2015 by sub_grade
    int_rate_dict = {'A1':0.0603, 'A2':0.0649, 'A3':0.0699, 'A4':0.0749, 'A5':0.0819,
                     'B1':0.0867, 'B2':0.0949, 'B3':0.1049, 'B4':0.1144, 'B5':0.1199,
                     'C1':0.1239, 'C2':0.1299, 'C3':0.1366, 'C4':0.1431, 'C5':0.1499,
                     'D1':0.1559, 'D2':0.1599, 'D3':0.1649, 'D4':0.1714, 'D5':0.1786}
    

    # Calculating actual IRR for loans already matured with updated int_rate
    print "Calculating IRR..."

    IRR = actual_IRR(df, False, int_rate_dict)

    print IRR[:2]
    # dump_to_pickle(IRR, '../pickle/IRR_matured_actual.pkl')


if __name__ == '__main__':
    main_actual()
    # main_recent()