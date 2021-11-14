import pandas as pd
import json
import word_similarity_stat as ws
import ast
import numpy as np

df = pd.read_excel("./data_docs/enx_rel_disciplines_clean.xlsx", dtype= {"discipline": str, "id_list": list})
enx_disciplines = df['discipline'].to_list()
enx_id = df['id_list'].to_list()
enx_disciplines_dict = {a : b for a, b in zip(enx_disciplines, enx_id)}

biomedical_list = ast.literal_eval( enx_disciplines_dict.get('Biomedical'))
combustion_list = ast.literal_eval(enx_disciplines_dict.get('Combustion'))
computer_list = ast.literal_eval(enx_disciplines_dict.get('Computer'))
chemical_list = ast.literal_eval(enx_disciplines_dict.get('Chemical'))
aerospace_list = ast.literal_eval(enx_disciplines_dict.get('Aerospace'))

corpus_list = list(set(biomedical_list)^set(combustion_list)^set(computer_list)^set(chemical_list)^set(aerospace_list))

enx_list = {}
for a in biomedical_list:
    if a in corpus_list:
        enx_list[a] = 'Biomedical'
for a in combustion_list:
    if a in corpus_list:
        enx_list[a] = 'Combustion'
for a in computer_list:
    if a in corpus_list:
        enx_list[a] = 'Computer'
for a in chemical_list:
    if a in corpus_list:
        enx_list[a] = 'Chemical'
for a in aerospace_list:
    if a in corpus_list:
        enx_list[a] = 'Aerospace'


with open("./data_docs/enxModifiedDocs_2021_03_30.json", encoding='utf-8') as json_file:
    enx_rank_corpus =  json.load(json_file)
     
biomedical_list = [{'_key':doc.get('key'),
                    'discipline': enx_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['Biomedical'], 0, 'elib_model_s_100_e_5_w_5')
                    } for doc in enx_rank_corpus if doc.get('key') in enx_list]

combustion_list = [{'_key':doc.get('key'),
                    'discipline': enx_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['Combustion'], 0, 'elib_model_s_100_e_5_w_5')
                    } for doc in enx_rank_corpus if doc.get('key') in enx_list]

computer_list =   [{'_key':doc.get('key'),
                    'discipline': enx_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['Computer'], 0, 'elib_model_s_100_e_5_w_5')
                    } for doc in enx_rank_corpus if doc.get('key') in enx_list]

aerospace_list =     [{'_key':doc.get('key'),
                    'discipline': enx_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['Aerospace'], 0, 'elib_model_s_100_e_5_w_5')
                    } for doc in enx_rank_corpus if doc.get('key') in enx_list]

chemical_list =      [{'_key':doc.get('key'),
                    'discipline': enx_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['Chemical'], 0, 'elib_model_s_100_e_5_w_5')
                    } for doc in enx_rank_corpus if doc.get('key') in enx_list]

biomedical_hist = []
for doc in biomedical_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')],bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    biomedical_hist.append({'_key':doc.get('_key'),
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


df = pd.DataFrame(biomedical_hist)
df.to_excel("./result/document_retrieval/enx_elib/enx_elib_biomedical_bin_count.xlsx", index=False)  

combustion_hist = []
for doc in combustion_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    combustion_hist.append({'_key':doc.get('_key'),
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
    
df = pd.DataFrame(combustion_hist)  
df.to_excel("./result/document_retrieval/enx_elib/enx_elib_combustion_bin_count.xlsx", index=False) 

computer_hist = []
for doc in computer_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    computer_hist.append({'_key':doc.get('_key'),
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
 

df = pd.DataFrame(computer_hist)
df.to_excel("./result/document_retrieval/enx_elib/enx_elib_computer_bin_count.xlsx", index=False) 

aerospace_hist = []
for doc in aerospace_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    aerospace_hist.append({'_key':doc.get('_key'),
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
    
df = pd.DataFrame(aerospace_hist)
df.to_excel("./result/document_retrieval/enx_elib/enx_elib_aerospace_bin_count.xlsx", index=False) 

chemical_hist = []
for doc in chemical_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    chemical_hist.append({'_key':doc.get('_key'),
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
    
df = pd.DataFrame(chemical_hist)
df.to_excel("./result/document_retrieval/enx_elib/enx_elib_chemical_bin_count.xlsx", index=False)  
