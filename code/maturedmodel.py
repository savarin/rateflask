import numpy as np
import pandas as pd
import pickle


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
        monthly_payments[i] = (calc_monthly_payments(100, int_rate, 3)\
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


def get_payout_actual(X, date_range_length):
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
            payout_actual_x = [1] * x[1]
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
    payout_actual = get_payout_actual(X, date_range_length)
    monthly_payments = get_monthly_payments(X_int_rate, date_range_length)
    compound_curve = get_compound_curve(X_int_rate, date_range_length)

    actual_cashflows = []

    for i in xrange(len(payout_actual)):
        cashflow = payout_actual[i] * monthly_payments[i] * compound_curve[i]
        actual_cashflows.append(cashflow)

    return np.array(actual_cashflows)


def matured_IRR(df, actual_rate=True, rate_dict={}, date_range_length):
    '''
    Calculates IRR for loans that have already matured.
    '''
    df_select = df[['id', 'installment']]

    # Residual amount for current loans and those in grace period under
    # consideration are sufficiently small and considered fully paid
    df_select['default_status'] = df['loan_status'].map({
                                        'Fully Paid': 0, 
                                        'Does not meet the credit policy.  Status:Fully Paid': 0, 
                                        'Current': 0,
                                        'In Grace Period': 0,
                                        'Charged Off': 1, 
                                        'Does not meet the credit policy.  Status:Charged Off': 1,
                                        'Late (31-120 days)': 1})

    df_select['total_rec_pymnt'] = df['total_rec_prncp'] + df['total_rec_int']
    df_select['total_at_maturity'] = df['total_rec_late_fee'] + df['recoveries'] \
                                     - df['collection_recovery_fee']

    # Calculates the number of months installment fully paid
    df_select['months_paid'] = df_select[['total_rec_pymnt', 'installment']] \
                                .apply(lambda x: np.floor(x['total_rec_pymnt'] \
                                                 / x['installment']), axis=1)

    # Calculates the fraction of installment paid on final month
    df_select['residual'] = df_select[['total_rec_pymnt', 'installment', 'months_paid']] \
                            .apply(lambda x: (x['total_rec_pymnt'] / x['installment']) \
                                             - x['months_paid'], axis=1)

    # Calculates the amount received from recovery, assumed to be at maturity
    # and quoted as a multiple of the monthly installment
    df_select['recovery'] = df_select[['total_at_maturity', 'installment']] \
                            .apply(lambda x: (x['total_at_maturity'] \
                                              / x['installment']), axis=1)

    X = df_select[['default_status', 'months_paid', 'residual', 'recovery']].values

    if actual_rate:
        X_int_rate = df['int_rate'].map(lambda x: float(str(x).strip('%')) / 100).values
    else:
        X_int_rate = df['sub_grade'].map(rate_dict).values


    actual_cashflows = get_actual_cashflows(X, X_int_rate, date_range_length)


def main():
    df_3a = pd.read_csv('../data/LoanStats3a_securev1.csv', header=True).iloc[:-2, :]

    df = df_3a.copy()
    df = df[df['term'].str.contains('36', na=False)]
    df = df[df['grade'].isin(['A', 'B', 'C', 'D'])]
    
    df = df[(df['issue_d'].str.contains('2009')) \
         | (df['issue_d'].str.contains('2010')) \
         | (df['issue_d'].str.contains('2011'))]

    IRR = matured_IRR(df)


if __name__ == '__main__':
    main()