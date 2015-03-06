
from lendingclub import LendingClub
from lendingclub.filters import Filter
from pymongo import MongoClient, Connection
import psycopg2
import time


def request_loan_data(club, filter_search):
    '''
    Requests list of loans that can be invested in, then makes individual call
    for details of the loans
    '''
    print "Requesting loans..." 
    loan_results = club.search(filter_search, start_index=0, limit=1000)
    loans = loan_results['loans']
    loan_ids = [loan['loan_id'] for loan in loans]

    loan_details = []
    for loan_id in loan_ids:
        print " loan_id", loan_id
        request = club.session.get('/browse/loanDetailAj.action', query={'loan_id': loan_id})
        loan_details.append(request.json())
        time.sleep(1)

    return loan_results, loan_details


def main():
    print "Initializing APIs..."
    club = LendingClub()
    club.authenticate()

    client = MongoClient()
    
    db = client.lending    
    collection_search = db.collection_search
    collection_get = db.collection_get

    filter_search = Filter({'exclude_existing': False,
                            'funding_progress': 0,
                            'grades': {'All': False,
                                       'A': True,
                                       'B': True,
                                       'C': True,
                                       'D': True,
                                       'E': False,
                                       'F': False,
                                       'G': False},
                            'term': {'Year3': True, 'Year5': False}})

    loan_results, loan_details = request_loan_data(club, filter_search)

    collection_search.insert(loan_results)
    collection_get.insert(loan_details)



    # PROCESS DATA
    # PROCESS DATA
    # PROCESS DATA


    # trends - simply list of length 10

    # conn = psycopg2.connect(dbname='lend', user='postgres', host='/tmp')
    # c = conn.cursor()

    # c.execute(
    #     ''' 
    #     INSERT INTO trends
    #     VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {});
    #     '''.format(*trends)
    # )

    # conn.commit()
    # conn.close()


if __name__ == '__main__':
    main()