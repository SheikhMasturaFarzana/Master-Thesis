from rank_bm25 import BM25Okapi

def get_docs(corpus, query, limit):
    
    """ returning dataframe containing ids of top documents
    Args:
        corpus: list of dicts with documents texts and keys
        queries: list of queries
        limit: number of hits per query according to original labelling
        df_name: dataframe name
        
    Returns:
        (dataframe) - each row contains list of documents that match the query
    """ 
    corpus_keys = [doc['_key'] for doc in corpus]
    tokenized_corpus = [doc['searchtext'].lower().split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split(" ") 
    doc_scores = bm25.get_scores(tokenized_query)
    
    score_dict = []
    for key, score in zip (corpus_keys, doc_scores):
        score_dict.append({'_key':key, 'score':score})
        
    score_dict = sorted(score_dict, key=lambda k: k['score'], reverse = True) 
    
    return [item['_key'] for item in score_dict][:limit]
    