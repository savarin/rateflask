import numpy as np
import pandas as pd


def process_payment(df_raw):
    '''
    Process payment information pertaining to matured loans.
    '''
    grade_mask = df_raw['grade'].isin(['A', 'B', 'C', 'D'])
    term_mask = df_raw['term'].str.contains('36', na=False)
    date_mask = (df_raw['issue_d'].str.contains('2009', na=False)) \
                | (df_raw['issue_d'].str.contains('2010', na=False)) \
                | (df_raw['issue_d'].str.contains('2011', na=False))

    df_raw = df_raw[grade_mask & term_mask & date_mask]

    df = df_raw[['id', 'sub_grade', 'installment']]
    df['int_rate'] = df_raw['int_rate'].map(lambda x: float(str(x).strip('%')) / 100).values

    # Residual amount for current loans and those in grace period under
    # consideration are sufficiently small and considered fully paid
    df['default_status'] = df_raw['loan_status'].map({
                                        'Fully Paid': 0, 
                                        'Does not meet the credit policy.  Status:Fully Paid': 0, 
                                        'Current': 0,
                                        'In Grace Period': 0,
                                        'Charged Off': 1, 
                                        'Does not meet the credit policy.  Status:Charged Off': 1,
                                        'Late (31-120 days)': 1})

    df['total_rec_pymnt'] = df_raw['total_rec_prncp'] + df_raw['total_rec_int']
    df['total_at_maturity'] = df_raw['total_rec_late_fee'] + df_raw['recoveries'] \
                                     - df_raw['collection_recovery_fee']

    # Calculates the number of months installment fully paid
    df['months_paid'] = df[['total_rec_pymnt', 'installment']] \
                                .apply(lambda x: np.floor(x['total_rec_pymnt'] \
                                                 / x['installment']), axis=1)

    # Calculates the fraction of installment paid on final month
    df['residual'] = df[['total_rec_pymnt', 'installment', 'months_paid']] \
                            .apply(lambda x: (x['total_rec_pymnt'] / x['installment']) \
                                             - x['months_paid'], axis=1)

    # Calculates the amount received from recovery, assumed to be at maturity
    # and quoted as a multiple of the monthly installment
    df['recovery'] = df[['total_at_maturity', 'installment']] \
                            .apply(lambda x: (x['total_at_maturity'] \
                                              / x['installment']), axis=1)

    return df[['sub_grade', 'int_rate', 
               'default_status', 'months_paid', 'residual', 'recovery']]