import pandas as pd
import json
import word_similarity_stat as ws
import ast
import numpy as np

df = (pd.read_excel(".\data_docs\esr_rel_all.xlsx")).drop('_key', 1)
esr_disciplines_dict = dict(df.values)

deep_learning_list = ast.literal_eval( esr_disciplines_dict.get('deep learning'))
question_answering_list = ast.literal_eval(esr_disciplines_dict.get('question answering'))
computer_vision_list = ast.literal_eval(esr_disciplines_dict.get('computer vision'))
cryptography_list = ast.literal_eval(esr_disciplines_dict.get('cryptography'))
information_geometry_list = ast.literal_eval(esr_disciplines_dict.get('information geometry'))

corpus_list = list(set(deep_learning_list)^set(question_answering_list)^set(computer_vision_list)^set(cryptography_list)^set(information_geometry_list))

esr_list = {}
for a in deep_learning_list:
    if a in corpus_list:
        esr_list[a] = 'deep learning'
for a in question_answering_list:
    if a in corpus_list:
        esr_list[a] = 'question answering'
for a in computer_vision_list:
    if a in corpus_list:
        esr_list[a] = 'computer vision'
for a in cryptography_list:
    if a in corpus_list:
        esr_list[a] = 'cryptography'
for a in information_geometry_list:
    if a in corpus_list:
        esr_list[a] = 'information geometry'


with open(".\data_docs\esr_pidf_ranks.json", encoding='utf-8') as json_file:
    esr_rank_corpus =  json.load(json_file)
     
deep_learning_list = [{'_key':doc.get('key'),
                    'discipline': esr_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['deep-learning'], 0, 'nb_vec')
                    } for doc in esr_rank_corpus if doc.get('key') in esr_list]

question_answering_list = [{'_key':doc.get('key'),
                    'discipline': esr_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['question-answering'], 0, 'nb_vec')
                    } for doc in esr_rank_corpus if doc.get('key') in esr_list]

computer_vision_list =   [{'_key':doc.get('key'),
                    'discipline': esr_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['computer-vision'], 0, 'nb_vec')
                    } for doc in esr_rank_corpus if doc.get('key') in esr_list]

information_geometry_list =     [{'_key':doc.get('key'),
                    'discipline': esr_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['information-geometry'], 0, 'nb_vec')
                    } for doc in esr_rank_corpus if doc.get('key') in esr_list]

cryptography_list =      [{'_key':doc.get('key'),
                    'discipline': esr_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['cryptography'], 0, 'nb_vec')
                    } for doc in esr_rank_corpus if doc.get('key') in esr_list]

deep_learning_hist = []
for doc in deep_learning_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')],bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    deep_learning_hist.append({'_key':doc.get('_key'),
                            'discipline': doc.get('discipline'),
                            '0.0-0.1':"{:.2f}".format(hist[0]/sum(hist)),
                            '0.1-0.2':"{:.2f}".format(hist[1]/sum(hist)),
                            '0.2-0.3':"{:.2f}".format(hist[2]/sum(hist)),
                            '0.3-0.4':"{:.2f}".format(hist[3]/sum(hist)),
                            '0.4-0.5':"{:.2f}".format(hist[4]/sum(hist)),
                            '0.5-0.6':"{:.2f}".format(hist[5]/sum(hist)),
                            '0.6-0.7':"{:.2f}".format(hist[6]/sum(hist)),
                            '0.7-0.8':"{:.2f}".format(hist[7]/sum(hist)),
                            '0.8-0.9':"{:.2f}".format(hist[8]/sum(hist)),
                            '0.9-1.0':"{:.2f}".format(hist[9]/sum(hist))
                            })


df = pd.DataFrame(deep_learning_hist)
df.to_excel("./result/document_retrieval/esr_nb/esr_nb_deep_learning_bin_count.xlsx", index=False)  

question_answering_hist = []
for doc in question_answering_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    question_answering_hist.append({'_key':doc.get('_key'),
                            'discipline': doc.get('discipline'),
                            '0.0-0.1':"{:.2f}".format(hist[0]/sum(hist)),
                            '0.1-0.2':"{:.2f}".format(hist[1]/sum(hist)),
                            '0.2-0.3':"{:.2f}".format(hist[2]/sum(hist)),
                            '0.3-0.4':"{:.2f}".format(hist[3]/sum(hist)),
                            '0.4-0.5':"{:.2f}".format(hist[4]/sum(hist)),
                            '0.5-0.6':"{:.2f}".format(hist[5]/sum(hist)),
                            '0.6-0.7':"{:.2f}".format(hist[6]/sum(hist)),
                            '0.7-0.8':"{:.2f}".format(hist[7]/sum(hist)),
                            '0.8-0.9':"{:.2f}".format(hist[8]/sum(hist)),
                            '0.9-1.0':"{:.2f}".format(hist[9]/sum(hist))
                            })
    
df = pd.DataFrame(question_answering_hist)  
df.to_excel("./result/document_retrieval/esr_nb/esr_nb_question_answering_bin_count.xlsx", index=False) 

computer_vision_hist = []
for doc in computer_vision_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    computer_vision_hist.append({'_key':doc.get('_key'),
                          'discipline': doc.get('discipline'),
                          '0.0-0.1':"{:.2f}".format(hist[0]/sum(hist)),
                          '0.1-0.2':"{:.2f}".format(hist[1]/sum(hist)),
                          '0.2-0.3':"{:.2f}".format(hist[2]/sum(hist)),
                          '0.3-0.4':"{:.2f}".format(hist[3]/sum(hist)),
                          '0.4-0.5':"{:.2f}".format(hist[4]/sum(hist)),
                          '0.5-0.6':"{:.2f}".format(hist[5]/sum(hist)),
                          '0.6-0.7':"{:.2f}".format(hist[6]/sum(hist)),
                          '0.7-0.8':"{:.2f}".format(hist[7]/sum(hist)),
                          '0.8-0.9':"{:.2f}".format(hist[8]/sum(hist)),
                          '0.9-1.0':"{:.2f}".format(hist[9]/sum(hist))
                          })
 

df = pd.DataFrame(computer_vision_hist)
df.to_excel("./result/document_retrieval/esr_nb/esr_nb_computer_vision_bin_count.xlsx", index=False) 

information_geometry_hist = []
for doc in information_geometry_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    information_geometry_hist.append({'_key':doc.get('_key'),
                          'discipline': doc.get('discipline'),
                          '0.0-0.1':"{:.2f}".format(hist[0]/sum(hist)),
                          '0.1-0.2':"{:.2f}".format(hist[1]/sum(hist)),
                          '0.2-0.3':"{:.2f}".format(hist[2]/sum(hist)),
                          '0.3-0.4':"{:.2f}".format(hist[3]/sum(hist)),
                          '0.4-0.5':"{:.2f}".format(hist[4]/sum(hist)),
                          '0.5-0.6':"{:.2f}".format(hist[5]/sum(hist)),
                          '0.6-0.7':"{:.2f}".format(hist[6]/sum(hist)),
                          '0.7-0.8':"{:.2f}".format(hist[7]/sum(hist)),
                          '0.8-0.9':"{:.2f}".format(hist[8]/sum(hist)),
                          '0.9-1.0':"{:.2f}".format(hist[9]/sum(hist))
                          })
    
df = pd.DataFrame(information_geometry_hist)
df.to_excel("./result/document_retrieval/esr_nb/esr_nb_information_geometry_bin_count.xlsx", index=False) 

cryptography_hist = []
for doc in cryptography_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    cryptography_hist.append({'_key':doc.get('_key'),
                        'discipline': doc.get('discipline'),
                        '0.0-0.1':"{:.2f}".format(hist[0]/sum(hist)),
                        '0.1-0.2':"{:.2f}".format(hist[1]/sum(hist)),
                        '0.2-0.3':"{:.2f}".format(hist[2]/sum(hist)),
                        '0.3-0.4':"{:.2f}".format(hist[3]/sum(hist)),
                        '0.4-0.5':"{:.2f}".format(hist[4]/sum(hist)),
                        '0.5-0.6':"{:.2f}".format(hist[5]/sum(hist)),
                        '0.6-0.7':"{:.2f}".format(hist[6]/sum(hist)),
                        '0.7-0.8':"{:.2f}".format(hist[7]/sum(hist)),
                        '0.8-0.9':"{:.2f}".format(hist[8]/sum(hist)),
                        '0.9-1.0':"{:.2f}".format(hist[9]/sum(hist))
                        })
    
df = pd.DataFrame(cryptography_hist)
df.to_excel("./result/document_retrieval/esr_nb/esr_nb_cryptography_bin_count.xlsx", index=False)  


