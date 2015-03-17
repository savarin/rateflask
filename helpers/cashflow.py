import numpy as np


def calc_monthly_payment(loan_amnt, int_rate, term):
    '''
    Calculates monthly payments (principal + interest) for loan with specified
    term and interest rate, with a declining principal as per amortizing schedule.

    Parameters:
    loan_amnt: Principal amount for loan. integer.
    int_rate: Monthly interest rate of loan. float.
    term: Term length of loan. integer.

    Returns:
    Monthly payment due. float.
    '''
    monthly_rate = int_rate / 12
    date_range_length = term * 12

    # Closed form formula to determine periodic payment.
    # http://en.wikipedia.org/wiki/amortization_calculator
    numerator = monthly_rate * ((1 + monthly_rate) ** date_range_length)
    denominator = ((1 + monthly_rate) ** date_range_length) - 1

    return loan_amnt * numerator / denominator


def get_monthly_payments(X_int_rate, date_range_length):
    '''
    Generates cashflow of monthly payments for each loan, i.e. monthly payment
    due calculated in calc_monthly_payment in array of length date_range_length.

    X_int_rate: Interest rate paid by each loan in X, 1-dimensional. numpy array.
    date_range_length: Total length of calculation period, normally 36. integer.

    Returns:
    Cashflow of monthly payments. numpy array.
    '''
    monthly_payments = np.ones((X_int_rate.shape[0], date_range_length))
    term = date_range_length / 12

    for i, int_rate in enumerate(X_int_rate):
        monthly_payments[i] = (calc_monthly_payment(1, int_rate, term)
                               * monthly_payments[i])

    return monthly_payments


def get_compound_curve(X_compound_rate, date_range_length):
    '''
    Generates compounding curve for each loan, assumes coupon reinvested in
    investment of similar return.

    Parameters:
    X_compound_rate: Compounding rate for time value of money calculations, 
    1-dimensional. numpy array.
    date_range_length: Total length of calculation period, normally 36. integer.

    Returns:
    Multiplication factors that correspond to compounding multiple of each 
    period. numpy array.
    '''
    compound_curve = []

    for i, int_rate in enumerate(X_compound_rate):
        compound_curve.append(np.array([(1 + int_rate / 12)**(i-1) for i
                                        in xrange(date_range_length, 0, -1)]))

    return np.array(compound_curve)


def get_cashflows(payout_curve, X_int_rate, X_compound_rate, date_range_length):
    '''
    Generates cashflow for each loan, i.e. monthly payments multiplied by 
    probability (or actuality if loan has matured) of receiving that payment and 
    compounded to the maturity of the loan.

    Parameters:
    payout_curve: Probability of receiving specific payment if calculating 
    expected_IRR, or fraction of monthly payment paid if calculating actual_IRR.
    numpy array.
    X_int_rate: Interest rate paid by each loan in X, 1-dimensional. numpy array.
    X_compound_rate: Compounding rate for time value of money calculations, 
    1-dimensional. numpy array.
    date_range_length: Total length of calculation period, normally 36. integer.

    Returns:
    Cashflow of each loan, after term and risk adjustment. numpy array.
    '''
        
    monthly_payments = get_monthly_payments(X_int_rate, date_range_length)
    compound_curve = get_compound_curve(X_compound_rate, date_range_length)

    cashflow_array = []

    for i in xrange(len(payout_curve)):
        cashflow = payout_curve[i] * monthly_payments[i] * compound_curve[i]
        cashflow_array.append(cashflow)

    return np.array(cashflow_array)



def calc_IRR(cashflows, term):
    '''
    Calculates IRR from generated cashflows.

    Parameters:
    cashflows: Cashflow stream over loan period.
    term: Term length of loan. integer.

    Returns:
    IRR figure. float.
    '''
    return [np.sum(cashflow)**(1./term) - 1 for cashflow in cashflows]