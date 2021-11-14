"""
Get processed or raw subset of datasets
"""
import pandas as pd
import ast
from collections import Counter
import tecominer_preprocess as tp
import arango_get_items as ar

def get_corpus(dataset, processed = False):
    
    if dataset == 'ENX':
        enx_df = pd.read_excel(".\data_docs\enx_rel_disciplines_clean.xlsx")
        enx_dict = dict(enx_df.values)
        
        biomedical_list = list((Counter(ast.literal_eval(enx_dict['Biomedical']))
                                -Counter(ast.literal_eval(enx_dict['Combustion']))
                                -Counter(ast.literal_eval(enx_dict['Computer']))
                                -Counter(ast.literal_eval(enx_dict['Aerospace']))
                                -Counter(ast.literal_eval(enx_dict['Chemical']))
                                ).elements())
        
        combustion_list = list((Counter(ast.literal_eval(enx_dict['Combustion']))
                                -Counter(ast.literal_eval(enx_dict['Biomedical']))
                                -Counter(ast.literal_eval(enx_dict['Computer']))
                                -Counter(ast.literal_eval(enx_dict['Aerospace']))
                                -Counter(ast.literal_eval(enx_dict['Chemical']))
                                ).elements())
        
        computer_list = list((Counter(ast.literal_eval(enx_dict['Computer']))
                                -Counter(ast.literal_eval(enx_dict['Biomedical']))
                                -Counter(ast.literal_eval(enx_dict['Combustion']))
                                -Counter(ast.literal_eval(enx_dict['Aerospace']))
                                -Counter(ast.literal_eval(enx_dict['Chemical']))
                                ).elements())
        
        aerospace_list = list((Counter(ast.literal_eval(enx_dict['Aerospace']))
                                -Counter(ast.literal_eval(enx_dict['Biomedical']))
                                -Counter(ast.literal_eval(enx_dict['Combustion']))
                                -Counter(ast.literal_eval(enx_dict['Computer']))
                                -Counter(ast.literal_eval(enx_dict['Chemical']))
                                ).elements())
        
        chemical_list = list((Counter(ast.literal_eval(enx_dict['Chemical']))
                                -Counter(ast.literal_eval(enx_dict['Biomedical']))
                                -Counter(ast.literal_eval(enx_dict['Combustion']))
                                -Counter(ast.literal_eval(enx_dict['Computer']))
                                -Counter(ast.literal_eval(enx_dict['Aerospace']))
                                ).elements())
        
        enx_corpus = ar.get_items('ENX_DATASET', 'documents_og', 'searchtext')
        
        enx_corpus_subset = [doc for doc in enx_corpus if doc.get('_key') in 
                             (biomedical_list
                              +combustion_list
                              +computer_list
                              +aerospace_list
                              +chemical_list)
                             ]
        
        if processed :
            enx_corpus_processed=[]
            [enx_corpus_processed.append({'_key':text.get('_key'),
                                  'searchtext':tp.clean_text(text.get('searchtext'),
                                  pos = ["ADJ","ADP","ADV", "AUX", "INTJ", "CONJ", "NOUN", 
                                         "DET","PROPN","VERB","PART","PRON","SCONJ","X"])}
                             )for text in enx_corpus_subset]
            
            enx_corpus_subset = enx_corpus_processed
            
        return {'corpus':enx_corpus_subset, 
                 'biomedical_list':biomedical_list, 
                 'combustion_list':combustion_list, 
                 'computer_list': computer_list, 
                 'aerospace_list': aerospace_list,
                 'chemical_list':chemical_list
                 }
    
    elif dataset == 'ESR':
        esr_df = (pd.read_excel(".\data_docs\esr_rel_all.xlsx")).drop('_key', 1)
        esr_dict = dict(esr_df.values)
        
        deep_learning_list = list((Counter(ast.literal_eval(esr_dict['deep learning']))
                                -Counter(ast.literal_eval(esr_dict['question answering']))
                                -Counter(ast.literal_eval(esr_dict['computer vision']))
                                -Counter(ast.literal_eval(esr_dict['information geometry']))
                                -Counter(ast.literal_eval(esr_dict['cryptography']))
                                ).elements())
        
        question_answering_list = list((Counter(ast.literal_eval(esr_dict['question answering']))
                                -Counter(ast.literal_eval(esr_dict['deep learning']))
                                -Counter(ast.literal_eval(esr_dict['computer vision']))
                                -Counter(ast.literal_eval(esr_dict['information geometry']))
                                -Counter(ast.literal_eval(esr_dict['cryptography']))
                                ).elements())
        
        computer_vision_list = list((Counter(ast.literal_eval(esr_dict['computer vision']))
                                -Counter(ast.literal_eval(esr_dict['deep learning']))
                                -Counter(ast.literal_eval(esr_dict['question answering']))
                                -Counter(ast.literal_eval(esr_dict['information geometry']))
                                -Counter(ast.literal_eval(esr_dict['cryptography']))
                                ).elements())
        
        information_geometry_list = list((Counter(ast.literal_eval(esr_dict['information geometry']))
                                -Counter(ast.literal_eval(esr_dict['deep learning']))
                                -Counter(ast.literal_eval(esr_dict['question answering']))
                                -Counter(ast.literal_eval(esr_dict['computer vision']))
                                -Counter(ast.literal_eval(esr_dict['cryptography']))
                                ).elements())
        
        cryptography_list = list((Counter(ast.literal_eval(esr_dict['cryptography']))
                                -Counter(ast.literal_eval(esr_dict['deep learning']))
                                -Counter(ast.literal_eval(esr_dict['question answering']))
                                -Counter(ast.literal_eval(esr_dict['computer vision']))
                                -Counter(ast.literal_eval(esr_dict['information geometry']))
                                ).elements())
        
        esr_corpus = ar.get_items('ESR_DATASET', 'documents_alt', 'searchtext')
        esr_corpus_subset = [doc for doc in esr_corpus if doc.get('_key') in 
                             (deep_learning_list
                              +question_answering_list
                              +computer_vision_list
                              +information_geometry_list
                              +cryptography_list)
                             ]
        
        if processed:
            esr_corpus_processed=[]
            [esr_corpus_processed.append({'_key':text.get('_key'),
                                  'searchtext':tp.clean_text(text.get('searchtext'),
                                  pos = ["ADJ","ADP","ADV", "AUX", "INTJ", "CONJ", "NOUN", 
                                         "DET","PROPN","VERB","PART","PRON","SCONJ","X"])}
                             )for text in esr_corpus_subset]
            
            esr_corpus_subset = esr_corpus_processed
        
        return {'corpus':esr_corpus_subset,
                'deep_learning_list':deep_learning_list,
                'question_answering_list': question_answering_list,
                'computer_vision_list':computer_vision_list,
                'information_geometry_list':information_geometry_list,
                'cryptography_list':cryptography_list
                }
                 
    elif dataset == 'ENV':
        env_df = pd.read_excel(".\data_docs\env_rel.xlsx")
        env_dict = dict(env_df.values)
        
        energy_list = list((Counter(ast.literal_eval(env_dict['energy']))
                                -Counter(ast.literal_eval(env_dict['biodiversity']))
                                -Counter(ast.literal_eval(env_dict['soil']))
                                -Counter(ast.literal_eval(env_dict['agriculture']))
                                -Counter(ast.literal_eval(env_dict['chemicals']))
                                ).elements())
        
        biodiversity_list = list((Counter(ast.literal_eval(env_dict['biodiversity']))
                                -Counter(ast.literal_eval(env_dict['energy']))
                                -Counter(ast.literal_eval(env_dict['soil']))
                                -Counter(ast.literal_eval(env_dict['agriculture']))
                                -Counter(ast.literal_eval(env_dict['chemicals']))
                                ).elements())
        
        soil_list = list((Counter(ast.literal_eval(env_dict['soil']))
                                -Counter(ast.literal_eval(env_dict['energy']))
                                -Counter(ast.literal_eval(env_dict['biodiversity']))
                                -Counter(ast.literal_eval(env_dict['agriculture']))
                                -Counter(ast.literal_eval(env_dict['chemicals']))
                                ).elements())
        
        agriculture_list = list((Counter(ast.literal_eval(env_dict['agriculture']))
                                -Counter(ast.literal_eval(env_dict['energy']))
                                -Counter(ast.literal_eval(env_dict['biodiversity']))
                                -Counter(ast.literal_eval(env_dict['soil']))
                                -Counter(ast.literal_eval(env_dict['chemicals']))
                                ).elements())
        
        chemicals_list = list((Counter(ast.literal_eval(env_dict['chemicals']))
                                -Counter(ast.literal_eval(env_dict['energy']))
                                -Counter(ast.literal_eval(env_dict['biodiversity']))
                                -Counter(ast.literal_eval(env_dict['soil']))
                                -Counter(ast.literal_eval(env_dict['agriculture']))
                                ).elements())
        
        env_corpus = ar.get_items('ENV_DATASET', 'documents', 'searchtext')
        env_corpus_subset = [doc for doc in env_corpus if doc.get('_key') in 
                             (energy_list
                              +biodiversity_list
                              +soil_list
                              +agriculture_list
                              +chemicals_list)
                             ]
        if processed:
            env_corpus_processed=[]
            [env_corpus_processed.append({'_key':text.get('_key'),
                                  'searchtext':tp.clean_text(text.get('searchtext'),
                                  pos = ["ADJ","ADP","ADV", "AUX", "INTJ", "CONJ", "NOUN", 
                                         "DET","PROPN","VERB","PART","PRON","SCONJ","X"])}
                             )for text in env_corpus_subset]
            
            env_corpus_subset = env_corpus_processed
        
        return {'corpus': env_corpus_subset,
                'energy_list':energy_list,
                'biodiversity_list':biodiversity_list,
                'soil_list':soil_list,
                'agriculture_list':agriculture_list,
                'chemicals_list':chemicals_list
            }