import numpy as np
import pandas as pd
from helpers.preprocessing import process_payment
from helpers.cashflow import get_cashflows, calc_IRR



def get_actual_payout(X, date_range_length):
    '''
    Generates actual payout of each loan. If loan pays in full returns array of
    ones of length date_range_length, otherwise returns arrays of ones for paid 
    months, zeros for unpaid with residual and recovery values filled in in
    respective months

    Parameters:
    X: Values pertaining to actual payment made. numpy array.
    date_range_length: Total length of calculation period, normally 36. integer.

    Returns:
    Actual payout, as a list of a fractions of the required monthly payment. 
    numpy array.
    '''
    payout_actual = []

    for i, x in enumerate(X):
        # If loan pays in full, returns array of ones
        if (x[0] == 0) or (x[1] >= date_range_length):
            payout_actual_x = [1] * date_range_length
            payout_actual.append(payout_actual_x)
        # Otherwise returns array of fraction of monthly payment
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

    Parameters:
    X: Values pertaining to actual payment made. numpy array.
    X_int_rate: Interest rate paid by each loan in X, 1-dimensional. numpy array.
    X_compound_rate: Compounding rate for time value of money calculations, 
    1-dimensional. numpy array.
    date_range_length: Total length of calculation period, normally 36. integer.

    Returns:
    Actual cashflows, as a list of numbers indicating actual payments made. 
    numpy array.    
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

    Parameters:
    df: Validation data with features reflecting payments made.
    actual_rate: Choice as to whether actual interest rate is to be used, or
    a custom entry. Custom interest rates are generally chosen to allow for
    comparison of loans of the same sub-grade over time. boolean
    rate_dict: Custom interest rate dictionary if actual_rate was True. Key-
    value pair of loan sub-grade and interest rate. dictionary.
    actual_as_compound: Choice as to whether actual interest rate is to be
    used for time value of money calculations. boolean.
    compound_rate: Custom interest rate if actual_as_compound was true. float.

    Returns:
    Actual IRR of each loan. list of floats.
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