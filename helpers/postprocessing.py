import numpy as np
import pandas as pd


def generate_for_charts(df_max):
    '''
    Generates input files to be used for charts on page.
    '''
    df_chart = pd.Series(data = [0.0] * 20,
                     index = ['A1', 'A2', 'A3', 'A4', 'A5', 
                              'B1', 'B2', 'B3', 'B4', 'B5', 
                              'C1', 'C2', 'C3', 'C4', 'C5', 
                              'D1', 'D2', 'D3', 'D4', 'D5'])

    df_chart.update(df_max)
    df_chart = pd.DataFrame({'subgrade':df_chart.index, 'rate':df_chart})

    df_chart.iloc[0:5,:].to_csv('static/chart/chart1.tsv', sep='\t', \
                                    columns=['subgrade', 'rate'], index=None)
    df_chart.iloc[5:10,:].to_csv('static/chart/chart2.tsv', sep='\t', \
                                    columns=['subgrade', 'rate'], index=None)
    df_chart.iloc[10:15,:].to_csv('static/chart/chart3.tsv', sep='\t', \
                                    columns=['subgrade', 'rate'], index=None)
    df_chart.iloc[15:20,:].to_csv('static/chart/chart4.tsv', sep='\t', \
                                    columns=['subgrade', 'rate'], index=None)


def reformat_for_display(df_display):
    '''
    Reformat details for data table.
    '''
    df_display['term'] = df_display['term'].map(lambda x: str(x) + ' mth')
    df_display['loan_amnt'] = df_display['loan_amnt'].map(lambda x: '$' \
                                                + str(x/1000) + ',' + str(x)[-3:])
    df_display['percent_fund'] = df_display['percent_fund'].map(lambda x: \
                                                str(round(x*100,0)))
    df_display['int_rate'] = df_display['int_rate'].map(lambda x: \
                                                str(round(x*100,2)) + '%')
    df_display['IRR'] = df_display['IRR'].map(lambda x: str(round(x*100,2)) + '%')
    df_display['percent_diff'] = df_display['percent_diff'].map(lambda x: \
                                                str(round(x*100,0)))

    return df_display