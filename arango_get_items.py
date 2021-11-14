from arango import ArangoClient
from arango.collection import StandardCollection
import arango as aq
from operator import itemgetter


def get_items(db_name, collection_name, name):
    """ Connect to arangoDB to get documents from a collection

    Args:
        db_name: database name
        collection_name: collection name
        name: key for requested item from collection
        
    Returns:
        (list of dictionaries) - keys and items
    """
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db(db_name, username= 'root', password='admin')
    query = 'For doc in '+collection_name+' return doc'
    aql = db.aql
    cursor = aql.execute(query)
    outcome = [res for res in cursor]
    outcome = [{'_key': item.get('_key'), name:item.get(name)} for item in outcome]
    return outcome

