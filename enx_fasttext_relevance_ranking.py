# -*- coding: utf-8 -*-
"""
This script compares elib and ft_en_cc extracted dictonaries with elib_word 
rank dictionaries and produces a specific rank of documents for different search words.
"""
import operator
import pandas as pd
import json
import corpus_subset 
import word_similarity_stat as ws

def get_global_average(corpus, fc_dict):
            
        sum_words=0
        sum_unique_words = 0
        count = 0
        for doc in fc_dict:
            searchtext = next((item for item in corpus if item["_key"] == doc.get('_key')), 0).get('searchtext',0)
            text_list = searchtext.split()
            count = count+1
            sum_words = sum_words+len(text_list)
            sum_unique_words = sum_unique_words + len(list(set(text_list)))
        return sum_unique_words/count, sum_words/count
    
    
def get_fX(k,x):
    return (x*(k+1))/(k+x)

def get_gY(a,b,y):
    return y/((1-b)+(b*a))

def get_rank_dict(fc_dict,rank_dict,rank_threshold,
                  word_dicts,words, filename,
                  p1_numer, p1_denom,
                  p2_numer, p2_denom):
    sum = 0
    factor_1 = 0
    for items in fc_dict:
        try:
           numer_1 = (((3*((items[1]-rank_threshold)**2))/(4*((1-rank_threshold)**2)))+(1/4))*rank_dict.get(items[0], 0)
           numer_2 = get_fX(2,(get_gY((p2_numer/p2_denom), 0.1, items[2])))
           denom = next(iter(rank_dict.values()))
           factor_1 += (numer_1*numer_2)/denom
           
        except:
            pass
    factor_2 = get_fX(2,(get_gY((p1_numer/p1_denom), 0.4, len(fc_dict))))
    sum = factor_1 * factor_2
    word_dicts[words].append((filename,sum))
    return word_dicts

def get_similarity(corpus, queries, threshold, model ):
    output = []
    for doc in corpus:
        f_dict = ws.evaluate_similarities_v1(doc.get('searchtext'), queries, threshold, model)
        if (f_dict!={}):
            output.append({'_key':doc.get('_key'), 'similarity_dict':f_dict})
    return output

#################  ENX   #################
## Chenge name of word embedding model, file directory and filenames
  
enx_corpus_dict = corpus_subset.get_corpus('ENX', 1)
enx_corpus = enx_corpus_dict['corpus']

corpus_list = [doc['_key'] for doc in enx_corpus]

with open(".\data_docs\enxModifiedDocs_2021_03_30.json", encoding='utf-8') as json_file:
    enx_rank_corpus =  json.load(json_file)

enx_rank = [doc for doc in enx_rank_corpus if doc.get('key') in corpus_list]

thresholds = [ 0.25,0.30, 0.35, 0.40, 0.50, 0.60]
for threshold in thresholds: 
    
    words = ["Biomedical", "Combustion", "Computer", "Aerospace", "Chemical"]
    similarity_dict = get_similarity(enx_corpus, words, threshold, 'elib_model_s_100_e_5_w_5')    
    p1_denom, p2_denom = get_global_average(enx_corpus, similarity_dict)

    rank_thresholds = [0,0.1,0.5]
    for rank_threshold in rank_thresholds:
        word_dicts = {x: [] for x in words}
        for doc in similarity_dict:
            fc_sim = doc.get('similarity_dict')
            rank_dict = dict(next((item for item in enx_rank if item["key"] == doc.get('_key')), 0).get('rank',0))
            doc_words = (next((item for item in enx_corpus if item["_key"] == doc.get('_key')), 0).get('searchtext',0)).split()
            
            p1_numer = len(list(set(doc_words)))
            p2_numer = len(doc_words)
            
            for i in range(len(words)):
                fc_dict = fc_sim.get(words[i])
    
                if (fc_dict!=None):
                    get_rank_dict(fc_dict,rank_dict,rank_threshold,
                                  word_dicts,words[i],
                                  doc.get('_key'),
                                  p1_numer, p1_denom,
                                  p2_numer, p2_denom)
          
        fasttext_dict = []
        for word in word_dicts:
            word_list = enx_corpus_dict.get(str(word.lower()+'_list'))
            fasttext_list = [a[0] for a in sorted( word_dicts.get(word), key=operator.itemgetter(1), reverse = True)[:(len(word_list)+3)]]
            fasttext_dict.append({'keyword': word,
                                  'true_hit': len(word_list),
                                  'common_hit': len(list(set(fasttext_list)&set(word_list))),
                                  'ft_hit': len(fasttext_list),
                                  'true_list': word_list,
                                  'common_list': list(set(fasttext_list)&set(word_list)),
                                  'ft_list': fasttext_list,
                                  })
        
        df = pd.DataFrame(fasttext_dict)
        df.to_excel("./result/relevance_ranking/enx_elib_threshold_eval/enx_elib_"+str(threshold)+"_"+str(rank_threshold)+".xlsx", index=False)   