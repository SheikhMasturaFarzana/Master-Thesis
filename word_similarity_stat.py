# -*- coding: utf-8 -*-
"""
This script uses a string of words and list of search words, each search word 
is compared with all the words from the string using one of the Fasttxet model.
If a word pair has similarity greater than a given threshold, it is stored in 
tuple along with the similarity value and the count of occurance of the word. 
This tuple is appended to a list which is appended to a dict and the search
word is used as the key.
"""


from operator import itemgetter
import requests

s = requests.Session()
def evaluate_similarities_v1(text, words, threshold, model):
    text_list = text.split()
    text_words = list(set(text_list))
    result_dict = {}
    for word in words:
        url = str('http://127.0.0.1:8000/fasttext/'+model+'/similaritylist?baseword='+word)
        res = s.post(url, json={"comparewords": text_words})
        text_sims = res.json()
        _words = [[p[0], p[1], text_list.count(p[0])] for p in zip(text_words, text_sims)]      
        _word = sorted((t for t in _words if (t[1])>threshold), key=itemgetter(1), reverse=True)   
        if(_word!=[]):
            result_dict[word]=_word
    return result_dict
            
def evaluate_similarities_v2(text_list, words, threshold, model):
    """ evaluates similarity of search words with text
    
    Args:
        (string) textlist = list containing tuples with text and rank
        (list) words = search words
        model  = model name 
        threshold = threshold of similarity 
        
    Returns:
        (dict) result_dict: dict of search words each containing a tuple of 
                            matched words, similarity and word count
    """
    
    text_words = [a[0] for a in text_list]
    text_rank = [a[1] for a in text_list]
    _word = []
    for word in words:
        url = str('http://127.0.0.1:8000/fasttext/'+model+'/similaritylist?baseword='+word)
        res = s.post(url, json={"comparewords": text_words})
        text_sims = res.json()
        _words = [[p[0], p[1], p[2]] for p in zip(text_words, text_sims, text_rank)]
        _word = sorted((t for t in _words if (t[1])>threshold), key=itemgetter(1), reverse=True)

    return _word





