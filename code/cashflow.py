import numpy as np


def calc_monthly_payments(loan_amnt, int_rate, term):
    '''
    Calculates monthly payments (principal + interest) for loan with specified
    term and interest rate
    '''
    monthly_rate = int_rate / 12
    date_range_length = term * 12

    numerator = monthly_rate * ((1 + monthly_rate) ** date_range_length)
    denominator = ((1 + monthly_rate) ** date_range_length) - 1

    return loan_amnt * numerator / denominator


def get_monthly_payments(X_int_rate, date_range_length):
    '''
    Generates monthly payments for each loan
    '''
    monthly_payments = np.ones((X_int_rate.shape[0], date_range_length))

    for i, int_rate in enumerate(X_int_rate):
        monthly_payments[i] = (calc_monthly_payments(1, int_rate, 3)
                               * monthly_payments[i])

    return monthly_payments


def get_compound_curve(X_int_rate, date_range_length):
    '''
    Generates compounding curve for each loan, assumes coupon reinvested in
    investment of similar return
    '''
    compound_curve = []

    for i, int_rate in enumerate(X_int_rate):
        compound_curve.append(np.array([(1 + int_rate / 12)**(i-1) for i
                                        in xrange(date_range_length, 0, -1)]))

    return np.array(compound_curve)


def get_cashflows(payout_curve, X_int_rate, date_range_length):
    '''
    Generates expected cashflow for each loan, i.e. monthly payments
    multiplied by probability of receiving that payment and compounded to
    the maturity of the loan
    '''
    monthly_payments = get_monthly_payments(X_int_rate, date_range_length)
    compound_curve = get_compound_curve(X_int_rate, date_range_length)

    cashflow_array = []

    for i in xrange(len(payout_curve)):
        cashflow = payout_curve[i] * monthly_payments[i] * compound_curve[i]
        cashflow_array.append(cashflow)

    return np.array(cashflow_array)