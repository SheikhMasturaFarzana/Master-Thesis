# -*- coding: utf-8 -*-
"""
Okapi BM25 Search 

"""

import pandas as pd
import ast
from collections import Counter
import arango_get_items as ar
import okapi_BM25 as bm25
import corpus_subset
#import tecominer_preprocess as tp

#################  ENX   #################
#avg_len 1272.63

enx_corpus_dict = corpus_subset.get_corpus('ENX')

enx_corpus_subset = enx_corpus_dict['corpus']
biomedical_list = enx_corpus_dict['biomedical_list']
combustion_list = enx_corpus_dict['combustion_list']
computer_list = enx_corpus_dict['computer_list']
aerospace_list = enx_corpus_dict['aerospace_list']
chemical_list = enx_corpus_dict['chemical_list']

enx_subset =  []

biomedical_hit = bm25.get_docs(enx_corpus_subset, 'Biomedical', len(biomedical_list)+5)
enx_subset.append({'keyword': 'biomedical' ,
                    'true_hit': len(biomedical_list),
                    'true_list': biomedical_list,
                    'common_hit':len(list(set(biomedical_list)&set(biomedical_hit))),
                    'common_list': list(set(biomedical_list)&set(biomedical_hit)), 
                    'bm25_hits': len(biomedical_hit),
                    'bm25_list': biomedical_hit,
                    })

combustion_hit = bm25.get_docs(enx_corpus_subset, 'Combustion', len(combustion_list)+5)
enx_subset.append({'keyword': 'combustion' ,
                    'true_hit': len(combustion_list),
                    'true_list': combustion_list,
                    'common_hit':len(list(set(combustion_list)&set(combustion_hit))),
                    'common_list': list(set(combustion_list)&set(combustion_hit)), 
                    'bm25_hits': len(combustion_hit),
                    'bm25_list': combustion_hit,
                    })

computer_hit = bm25.get_docs(enx_corpus_subset, 'Computer', len(computer_list)+5)
enx_subset.append({'keyword': 'computer' ,
                    'true_hit': len(computer_list),
                    'true_list': computer_list,
                    'common_hit':len(list(set(computer_list)&set(computer_hit))),
                    'common_list': list(set(computer_list)&set(computer_hit)), 
                    'bm25_hits': len(computer_hit),
                    'bm25_list': computer_hit,
                    })

aerospace_hit = bm25.get_docs(enx_corpus_subset, 'Aerospace', len(aerospace_list)+5)
enx_subset.append({'keyword': 'aerospace' ,
                    'true_hit': len(aerospace_list),
                    'true_list': aerospace_list,
                    'common_hit':len(list(set(aerospace_list)&set(aerospace_hit))),
                    'common_list': list(set(aerospace_list)&set(aerospace_hit)), 
                    'bm25_hits': len(aerospace_hit),
                    'bm25_list': aerospace_hit,
                    })

chemical_hit = bm25.get_docs(enx_corpus_subset, 'Chemical', len(chemical_list)+5)
enx_subset.append({'keyword': 'chemical' ,
                    'true_hit': len(chemical_list),
                    'true_list': chemical_list,
                    'common_hit':len(list(set(chemical_list)&set(chemical_hit))),
                    'common_list': list(set(chemical_list)&set(chemical_hit)), 
                    'bm25_hits': len(chemical_hit),
                    'bm25_list': chemical_hit,
                    })

enx_result = pd.DataFrame(enx_subset)
enx_result.to_excel("./result/bm25_search/enx_subset.xlsx", index=False) 


#################  ESR   #################
# avg_len = 926.16

esr_corpus_dict = corpus_subset.get_corpus('ESR')

esr_corpus_subset = esr_corpus_dict['corpus']
deep_learning_list = esr_corpus_dict['deep_learning_list']
question_answering_list = esr_corpus_dict['question_answering_list']
computer_vision_list = esr_corpus_dict['computer_vision_list']
information_geometry_list = esr_corpus_dict['information_geometry_list']
cryptography_list = esr_corpus_dict['cryptography_list']

esr_subset =  []

deep_learning_hit = bm25.get_docs(esr_corpus_subset, 'deep learning', len(deep_learning_list)+5)
esr_subset.append({'keyword': 'deep learning' ,
                    'true_hit': len(deep_learning_list),
                    'true_list': deep_learning_list,
                    'common_hit':len(list(set(deep_learning_list)&set(deep_learning_hit))),
                    'common_list': list(set(deep_learning_list)&set(deep_learning_hit)), 
                    'bm25_hits': len(deep_learning_hit),
                    'bm25_list': deep_learning_hit,
                    })

question_answering_hit = bm25.get_docs(esr_corpus_subset, 'question answering', len(question_answering_list)+5)
esr_subset.append({'keyword': 'question answering' ,
                    'true_hit': len(question_answering_list),
                    'true_list': question_answering_list,
                    'common_hit':len(list(set(question_answering_list)&set(question_answering_hit))),
                    'common_list': list(set(question_answering_list)&set(question_answering_hit)), 
                    'bm25_hits': len(question_answering_hit),
                    'bm25_list': question_answering_hit,
                    })

computer_vision_hit = bm25.get_docs(esr_corpus_subset, 'computer vision', len(computer_vision_list)+5)
esr_subset.append({'keyword': 'computer vision' ,
                    'true_hit': len(computer_vision_list),
                    'true_list': computer_vision_list,
                    'common_hit':len(list(set(computer_vision_list)&set(computer_vision_hit))),
                    'common_list': list(set(computer_vision_list)&set(computer_vision_hit)), 
                    'bm25_hits': len(computer_vision_hit),
                    'bm25_list': computer_vision_hit,
                    })

information_geometry_hit = bm25.get_docs(esr_corpus_subset, 'information geometry', len(information_geometry_list)+5)
esr_subset.append({'keyword': 'information geometry' ,
                    'true_hit': len(information_geometry_list),
                    'true_list': information_geometry_list,
                    'common_hit':len(list(set(information_geometry_list)&set(information_geometry_hit))),
                    'common_list': list(set(information_geometry_list)&set(information_geometry_hit)), 
                    'bm25_hits': len(information_geometry_hit),
                    'bm25_list': information_geometry_hit,
                    })

cryptography_hit = bm25.get_docs(esr_corpus_subset, 'cryptography', len(cryptography_list)+5)
esr_subset.append({'keyword': 'cryptography' ,
                    'true_hit': len(cryptography_list),
                    'true_list': cryptography_list,
                    'common_hit':len(list(set(cryptography_list)&set(cryptography_hit))),
                    'common_list': list(set(cryptography_list)&set(cryptography_hit)), 
                    'bm25_hits': len(cryptography_hit),
                    'bm25_list': cryptography_hit,
                    })

esr_result = pd.DataFrame(esr_subset)
esr_result.to_excel("./result/bm25_search/esr_subset.xlsx", index=False) 


#################  ENV   #################
#avg_len = 4146.87
env_corpus_dict = corpus_subset.get_corpus('ENV')

env_corpus_subset = env_corpus_dict['corpus']
energy_list = env_corpus_dict['energy_list']
biodiversity_list = env_corpus_dict['biodiversity_list']
soil_list = env_corpus_dict['soil_list']
agriculture_list = env_corpus_dict['agriculture_list']
chemicals_list = env_corpus_dict['chemicals_list']

env_subset =  []

energy_hit = bm25.get_docs(env_corpus_subset, 'energy', len(energy_list)+5)
env_subset.append({'keyword': 'energy' ,
                    'true_hit': len(energy_list),
                    'true_list': energy_list,
                    'common_hit':len(list(set(energy_list)&set(energy_hit))),
                    'common_list': list(set(energy_list)&set(energy_hit)), 
                    'bm25_hits': len(energy_hit),
                    'bm25_list': energy_hit,
                    })

biodiversity_hit = bm25.get_docs(env_corpus_subset, 'biodiversity', len(biodiversity_list)+5)
env_subset.append({'keyword': 'biodiversity' ,
                    'true_hit': len(biodiversity_list),
                    'true_list': biodiversity_list,
                    'common_hit':len(list(set(biodiversity_list)&set(biodiversity_hit))),
                    'common_list': list(set(biodiversity_list)&set(biodiversity_hit)), 
                    'bm25_hits': len(biodiversity_hit),
                    'bm25_list': biodiversity_hit,
                    })

soil_hit = bm25.get_docs(env_corpus_subset, 'soil', len(soil_list)+5)
env_subset.append({'keyword': 'soil' ,
                    'true_hit': len(soil_list),
                    'true_list': soil_list,
                    'common_hit':len(list(set(soil_list)&set(soil_hit))),
                    'common_list': list(set(soil_list)&set(soil_hit)), 
                    'bm25_hits': len(soil_hit),
                    'bm25_list': soil_hit,
                    })

agriculture_hit = bm25.get_docs(env_corpus_subset, 'agriculture', len(agriculture_list)+5)
env_subset.append({'keyword': 'agriculture' ,
                    'true_hit': len(agriculture_list),
                    'true_list': agriculture_list,
                    'common_hit':len(list(set(agriculture_list)&set(agriculture_hit))),
                    'common_list': list(set(agriculture_list)&set(agriculture_hit)), 
                    'bm25_hits': len(agriculture_hit),
                    'bm25_list': agriculture_hit,
                    })

chemicals_hit = bm25.get_docs(env_corpus_subset, 'chemicals', len(chemicals_list)+5)
env_subset.append({'keyword': 'chemicals' ,
                    'true_hit': len(chemicals_list),
                    'true_list': chemicals_list,
                    'common_hit':len(list(set(chemicals_list)&set(chemicals_hit))),
                    'common_list': list(set(chemicals_list)&set(chemicals_hit)), 
                    'bm25_hits': len(chemicals_hit),
                    'bm25_list': chemicals_hit,
                    })

env_result = pd.DataFrame(env_subset)
env_result.to_excel("./result/bm25_search/env_subset.xlsx", index=False)
