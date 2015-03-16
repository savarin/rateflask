from pymongo import MongoClient, Connection
from lendingclub import LendingClub
from lendingclub.filters import Filter
import time


def request_loan_data():
    '''
    Requests list of loans that can be invested in, then makes individual call
    for details of the loans. Results stored in MongoDB database.
    '''
    filter_search = {'exclude_existing': False,
                     'funding_progress': 0,
                     'grades': {'All': False,
                                'A': True,
                                'B': True,
                                'C': True,
                                'D': True,
                                'E': False,
                                'F': False,
                                'G': False},
                     'term': {'Year3': True, 'Year5': False}}
                 
    club = LendingClub()
    filter_search = Filter(filter_search)
    club.authenticate()

    loan_results = club.search(filter_search, start_index=0, limit=1000)
    loan_results = loan_results['loans']
    loan_ids = [loan['loan_id'] for loan in loan_results]

    loan_details = []
    for loan_id in loan_ids:
        print "loan_id", loan_id
        request = club.session.get('/browse/loanDetailAj.action', query={'loan_id': loan_id})
        loan_details.append(request.json())
        time.sleep(1)

    return loan_results, loan_details