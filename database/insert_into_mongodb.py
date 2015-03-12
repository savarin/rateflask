import pymongo


def insert_into_mongodb(loan_results, loan_details):
    '''
    Insert loan features and expected IRR into PostgresQL database.
    '''
    client = pymongo.MongoClient()
    db = client.rateflask   
    collection_search = db.collection_search
    collection_get = db.collection_get

    collection_search.insert(loan_results)
    collection_get.insert(loan_details)