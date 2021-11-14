"""
Keyword search on ENX, ENV and ESR Dataset

"""

import pandas as pd
import corpus_subset


def get_docs(docs, keyword):
    """ Get list of documents containing keyword

    Args:
        docs: list of dictionaries containing document key and searchtext
        keyword: query term
        
    Returns:
        (list of document keys) - list of documents containing keyword
    """
    doc_list = []
    for doc in docs:
        if keyword.lower() in doc['searchtext'].lower():
            doc_list.append(doc['_key'])
    return doc_list


#################  ENX   #################

enx_corpus_dict = corpus_subset.get_corpus('ENX')

enx_corpus_subset = enx_corpus_dict['corpus']
biomedical_list = enx_corpus_dict['biomedical_list']
combustion_list = enx_corpus_dict['combustion_list']
computer_list = enx_corpus_dict['computer_list']
aerospace_list = enx_corpus_dict['aerospace_list']
chemical_list = enx_corpus_dict['chemical_list']

enx_subset =  []

biomedical_hit = get_docs(enx_corpus_subset, 'Biomedical')
enx_subset.append({'keyword': 'biomedical' ,
                    'true_hit': len(biomedical_list),
                    'true_list': biomedical_list,
                    'common_hit':len(list(set(biomedical_list)&set(biomedical_hit))),
                    'common_list': list(set(biomedical_list)&set(biomedical_hit)), 
                    'search_hits': len(biomedical_hit),
                    'search_list': biomedical_hit,
                    })

combustion_hit = get_docs(enx_corpus_subset, 'Combustion')
enx_subset.append({'keyword': 'combustion' ,
                    'true_hit': len(combustion_list),
                    'true_list': combustion_list,
                    'common_hit':len(list(set(combustion_list)&set(combustion_hit))),
                    'common_list': list(set(combustion_list)&set(combustion_hit)), 
                    'search_hits': len(combustion_hit),
                    'search_list': combustion_hit,
                    })

computer_hit = get_docs(enx_corpus_subset, 'Computer')
enx_subset.append({'keyword': 'computer' ,
                    'true_hit': len(computer_list),
                    'true_list': computer_list,
                    'common_hit':len(list(set(computer_list)&set(computer_hit))),
                    'common_list': list(set(computer_list)&set(computer_hit)), 
                    'search_hits': len(computer_hit),
                    'search_list': computer_hit,
                    })

aerospace_hit = get_docs(enx_corpus_subset, 'Aerospace')
enx_subset.append({'keyword': 'aerospace' ,
                    'true_hit': len(aerospace_list),
                    'true_list': aerospace_list,
                    'common_hit':len(list(set(aerospace_list)&set(aerospace_hit))),
                    'common_list': list(set(aerospace_list)&set(aerospace_hit)), 
                    'search_hits': len(aerospace_hit),
                    'search_list': aerospace_hit,
                    })

chemical_hit = get_docs(enx_corpus_subset, 'Chemical')
enx_subset.append({'keyword': 'chemical' ,
                    'true_hit': len(chemical_list),
                    'true_list': chemical_list,
                    'common_hit':len(list(set(chemical_list)&set(chemical_hit))),
                    'common_list': list(set(chemical_list)&set(chemical_hit)), 
                    'search_hits': len(chemical_hit),
                    'search_list': chemical_hit,
                    })


enx_result = pd.DataFrame(enx_subset)
enx_result.to_excel("./result/keyword_search/enx_subset.xlsx", index=False) 

#################  ESR   #################

esr_corpus_dict = corpus_subset.get_corpus('ESR')

esr_corpus_subset = esr_corpus_dict['corpus']
deep_learning_list = esr_corpus_dict['deep_learning_list']
question_answering_list = esr_corpus_dict['question_answering_list']
computer_vision_list = esr_corpus_dict['computer_vision_list']
information_geometry_list = esr_corpus_dict['information_geometry_list']
cryptography_list = esr_corpus_dict['cryptography_list']

esr_subset =  []

deep_learning_hit = get_docs(esr_corpus_subset, 'deep learning')
esr_subset.append({'keyword': 'deep learning' ,
                    'true_hit': len(deep_learning_list),
                    'true_list': deep_learning_list,
                    'common_hit':len(list(set(deep_learning_list)&set(deep_learning_hit))),
                    'common_list': list(set(deep_learning_list)&set(deep_learning_hit)), 
                    'search_hits': len(deep_learning_hit),
                    'search_list': deep_learning_hit,
                    })

question_answering_hit = get_docs(esr_corpus_subset, 'question answering')
esr_subset.append({'keyword': 'question answering' ,
                    'true_hit': len(question_answering_list),
                    'true_list': question_answering_list,
                    'common_hit':len(list(set(question_answering_list)&set(question_answering_hit))),
                    'common_list': list(set(question_answering_list)&set(question_answering_hit)), 
                    'search_hits': len(question_answering_hit),
                    'search_list': question_answering_hit,
                    })

computer_vision_hit = get_docs(esr_corpus_subset, 'computer vision')
esr_subset.append({'keyword': 'computer vision' ,
                    'true_hit': len(computer_vision_list),
                    'true_list': computer_vision_list,
                    'common_hit':len(list(set(computer_vision_list)&set(computer_vision_hit))),
                    'common_list': list(set(computer_vision_list)&set(computer_vision_hit)), 
                    'search_hits': len(computer_vision_hit),
                    'search_list': computer_vision_hit,
                    })

information_geometry_hit = get_docs(esr_corpus_subset, 'information geometry')
esr_subset.append({'keyword': 'information geometry' ,
                    'true_hit': len(information_geometry_list),
                    'true_list': information_geometry_list,
                    'common_hit':len(list(set(information_geometry_list)&set(information_geometry_hit))),
                    'common_list': list(set(information_geometry_list)&set(information_geometry_hit)), 
                    'search_hits': len(information_geometry_hit),
                    'search_list': information_geometry_hit,
                    })

cryptography_hit = get_docs(esr_corpus_subset, 'cryptography')
esr_subset.append({'keyword': 'cryptography' ,
                    'true_hit': len(cryptography_list),
                    'true_list': cryptography_list,
                    'common_hit':len(list(set(cryptography_list)&set(cryptography_hit))),
                    'common_list': list(set(cryptography_list)&set(cryptography_hit)), 
                    'search_hits': len(cryptography_hit),
                    'search_list': cryptography_hit,
                    })

esr_result = pd.DataFrame(esr_subset)
esr_result.to_excel("./result/keyword_search/esr_subset.xlsx", index=False) 


#################  ENV   #################

env_corpus_dict = corpus_subset.get_corpus('ENV')

env_corpus_subset = env_corpus_dict['corpus']
energy_list = env_corpus_dict['energy_list']
biodiversity_list = env_corpus_dict['biodiversity_list']
soil_list = env_corpus_dict['soil_list']
agriculture_list = env_corpus_dict['agriculture_list']
chemicals_list = env_corpus_dict['chemicals_list']

env_subset =  []

energy_hit = get_docs(env_corpus_subset, 'energy')
env_subset.append({'keyword': 'energy' ,
                    'true_hit': len(energy_list),
                    'true_list': energy_list,
                    'common_hit':len(list(set(energy_list)&set(energy_hit))),
                    'common_list': list(set(energy_list)&set(energy_hit)), 
                    'search_hits': len(energy_hit),
                    'search_list': energy_hit,
                    })

biodiversity_hit = get_docs(env_corpus_subset, 'biodiversity')
env_subset.append({'keyword': 'biodiversity' ,
                    'true_hit': len(biodiversity_list),
                    'true_list': biodiversity_list,
                    'common_hit':len(list(set(biodiversity_list)&set(biodiversity_hit))),
                    'common_list': list(set(biodiversity_list)&set(biodiversity_hit)), 
                    'search_hits': len(biodiversity_hit),
                    'search_list': biodiversity_hit,
                    })

soil_hit = get_docs(env_corpus_subset, 'soil')
env_subset.append({'keyword': 'soil' ,
                    'true_hit': len(soil_list),
                    'true_list': soil_list,
                    'common_hit':len(list(set(soil_list)&set(soil_hit))),
                    'common_list': list(set(soil_list)&set(soil_hit)), 
                    'search_hits': len(soil_hit),
                    'search_list': soil_hit,
                    })

agriculture_hit = get_docs(env_corpus_subset, 'agriculture')
env_subset.append({'keyword': 'agriculture' ,
                    'true_hit': len(agriculture_list),
                    'true_list': agriculture_list,
                    'common_hit':len(list(set(agriculture_list)&set(agriculture_hit))),
                    'common_list': list(set(agriculture_list)&set(agriculture_hit)), 
                    'search_hits': len(agriculture_hit),
                    'search_list': agriculture_hit,
                    })

chemicals_hit = get_docs(env_corpus_subset, 'chemicals')
env_subset.append({'keyword': 'chemicals' ,
                    'true_hit': len(chemicals_list),
                    'true_list': chemicals_list,
                    'common_hit':len(list(set(chemicals_list)&set(chemicals_hit))),
                    'common_list': list(set(chemicals_list)&set(chemicals_hit)), 
                    'search_hits': len(chemicals_hit),
                    'search_list': chemicals_hit,
                    })

env_result = pd.DataFrame(env_subset)
env_result.to_excel("./result/keyword_search/env_subset.xlsx", index=False)
