import pandas as pd
import json
import word_similarity_stat as ws
import ast
import numpy as np

df = pd.read_excel("./data_docs/env_rel.xlsx", dtype= {"discipline": str, "id_list": list})
env_disciplines = df['query'].to_list()
env_id = df['id_list'].to_list()
env_disciplines_dict = {a : b for a, b in zip(env_disciplines, env_id)}

energy_list = ast.literal_eval( env_disciplines_dict.get('energy'))
biodiversity_list = ast.literal_eval(env_disciplines_dict.get('biodiversity'))
soil_list = ast.literal_eval(env_disciplines_dict.get('soil'))
chemicals_list = ast.literal_eval(env_disciplines_dict.get('chemicals'))
agriculture_list = ast.literal_eval(env_disciplines_dict.get('agriculture'))

corpus_list = list(set(energy_list)^set(biodiversity_list)^set(soil_list)^set(chemicals_list)^set(agriculture_list))

env_list = {}
for a in energy_list:
    if a in corpus_list:
        env_list[a] = 'energy'
for a in biodiversity_list:
    if a in corpus_list:
        env_list[a] = 'biodiversity'
for a in soil_list:
    if a in corpus_list:
        env_list[a] = 'soil'
for a in chemicals_list:
    if a in corpus_list:
        env_list[a] = 'chemicals'
for a in agriculture_list:
    if a in corpus_list:
        env_list[a] = 'agriculture'


with open("./data_docs/env_rankfile.json", encoding='utf-8') as json_file:
    env_rank_corpus =  json.load(json_file)
     
energy_list = [{'_key':doc.get('key'),
                    'discipline': env_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['energy'], 0, 'ft_en_cc')
                    } for doc in env_rank_corpus if doc.get('key') in env_list]

biodiversity_list = [{'_key':doc.get('key'),
                    'discipline': env_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['biodiversity'], 0, 'ft_en_cc')
                    } for doc in env_rank_corpus if doc.get('key') in env_list]

soil_list =   [{'_key':doc.get('key'),
                    'discipline': env_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['soil'], 0, 'ft_en_cc')
                    } for doc in env_rank_corpus if doc.get('key') in env_list]

agriculture_list =     [{'_key':doc.get('key'),
                    'discipline': env_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['agriculture'], 0, 'ft_en_cc')
                    } for doc in env_rank_corpus if doc.get('key') in env_list]

chemicals_list =      [{'_key':doc.get('key'),
                    'discipline': env_list.get(doc.get('key')),
                    'similarity_dict': ws.evaluate_similarities_v2(doc.get('rank'), ['chemicals'], 0, 'ft_en_cc')
                    } for doc in env_rank_corpus if doc.get('key') in env_list]

energy_hist = []
for doc in energy_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')],bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    energy_hist.append({'_key':doc.get('_key'),
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


df = pd.DataFrame(energy_hist)
df.to_excel("./result/document_retrieval/env_ft/env_ft_energy_bin_count.xlsx", index=False)  

biodiversity_hist = []
for doc in biodiversity_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    biodiversity_hist.append({'_key':doc.get('_key'),
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
    
df = pd.DataFrame(biodiversity_hist)  
df.to_excel("./result/document_retrieval/env_ft/env_ft_biodiversity_bin_count.xlsx", index=False) 

soil_hist = []
for doc in soil_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    soil_hist.append({'_key':doc.get('_key'),
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
 

df = pd.DataFrame(soil_hist)
df.to_excel("./result/document_retrieval/env_ft/env_ft_soil_bin_count.xlsx", index=False) 

agriculture_hist = []
for doc in agriculture_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    agriculture_hist.append({'_key':doc.get('_key'),
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
    
df = pd.DataFrame(agriculture_hist)
df.to_excel("./result/document_retrieval/env_ft/env_ft_agriculture_bin_count.xlsx", index=False) 

chemicals_hist = []
for doc in chemicals_list:
    hist, _ =  np.histogram([a[1] for a in doc.get('similarity_dict')], bins=10, range=(0, 1), weights = [a[2] for a in doc.get('similarity_dict')])
    chemicals_hist.append({'_key':doc.get('_key'),
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
    
df = pd.DataFrame(chemicals_hist)
df.to_excel("./result/document_retrieval/env_ft/env_ft_chemicals_bin_count.xlsx", index=False)  


