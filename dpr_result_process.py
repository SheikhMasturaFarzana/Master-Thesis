import json
import pandas as pd
import corpus_subset

esr_corpus_dict = corpus_subset.get_corpus('ESR',1) 
deep_learning_list = esr_corpus_dict['deep_learning_list'] 
question_answering_list = esr_corpus_dict ['question_answering_list'] 
computer_vision_list = esr_corpus_dict ['computer_vision_list'] 
information_geometry_list = esr_corpus_dict['information_geometry_list'] 
cryptography_list = esr_corpus_dict['cryptography_list'] 


dpr_deep_learning_data =json.load(open('./dpr_data/esr/esr_dpr_deep_learning.json', encoding = 'utf-8')) 
dpr_deep_learning_list =  [next(item for item in esr_corpus_dict['corpus'] if item['searchtext'] == doc).get('_key') for doc in dpr_deep_learning_data]

dpr_question_answering_data =json.load(open('./dpr_data/esr/esr_dpr_question_answering.json', encoding = 'utf-8')) 
dpr_question_answering_list =  [next(item for item in esr_corpus_dict['corpus'] if item['searchtext'] == doc).get('_key') for doc in dpr_question_answering_data]

dpr_computer_vision_data =json.load(open('./dpr_data/esr/esr_dpr_computer_vision.json', encoding = 'utf-8')) 
dpr_computer_vision_list =  [next(item for item in esr_corpus_dict['corpus'] if item['searchtext'] == doc).get('_key') for doc in dpr_computer_vision_data]

dpr_information_geometry_data =json.load(open('./dpr_data/esr/esr_dpr_information_geometry.json', encoding = 'utf-8')) 
dpr_information_geometry_list =  [next(item for item in esr_corpus_dict['corpus'] if item['searchtext'] == doc).get('_key') for doc in dpr_information_geometry_data]

dpr_cryptography_data =json.load(open('./dpr_data/esr/esr_dpr_cryptography.json', encoding = 'utf-8')) 
dpr_cryptography_list =  [next(item for item in esr_corpus_dict['corpus'] if item['searchtext'] == doc).get('_key') for doc in dpr_cryptography_data]

dpr_result = []

dpr_result.append({'keyword': 'deep learning' ,
                    'esr_hit': len(deep_learning_list),
                    'esr_list': deep_learning_list,
                    'common_hit':len(list(set(deep_learning_list)&set(dpr_deep_learning_list))),
                    'common_list': list(set(deep_learning_list)&set(dpr_deep_learning_list)), 
                    'dpr_hits': len(dpr_deep_learning_list),
                    'dpr_list': dpr_deep_learning_list,
                    })

dpr_result.append({'keyword': 'question answering' ,
                    'esr_hit': len(question_answering_list),
                    'esr_list': question_answering_list,
                    'common_hit':len(list(set(question_answering_list)&set(dpr_question_answering_list))),
                    'common_list': list(set(question_answering_list)&set(dpr_question_answering_list)), 
                    'dpr_hits': len(dpr_question_answering_list),
                    'dpr_list': dpr_question_answering_list,
                    })

dpr_result.append({'keyword': 'computer vision' ,
                    'esr_hit': len(computer_vision_list),
                    'esr_list': computer_vision_list,
                    'common_hit':len(list(set(computer_vision_list)&set(dpr_computer_vision_list))),
                    'common_list': list(set(computer_vision_list)&set(dpr_computer_vision_list)), 
                    'dpr_hits': len(dpr_computer_vision_list),
                    'dpr_list': dpr_computer_vision_list,
                    })

dpr_result.append({'keyword': 'information geometry' ,
                    'esr_hit': len(information_geometry_list),
                    'esr_list': information_geometry_list,
                    'common_hit':len(list(set(information_geometry_list)&set(dpr_information_geometry_list))),
                    'common_list': list(set(information_geometry_list)&set(dpr_information_geometry_list)), 
                    'dpr_hits': len(dpr_information_geometry_list),
                    'dpr_list': dpr_information_geometry_list,
                    })

dpr_result.append({'keyword': 'cryptography' ,
                    'esr_hit': len(cryptography_list),
                    'esr_list': cryptography_list,
                    'common_hit':len(list(set(cryptography_list)&set(dpr_cryptography_list))),
                    'common_list': list(set(cryptography_list)&set(dpr_cryptography_list)), 
                    'dpr_hits': len(dpr_cryptography_list),
                    'dpr_list': dpr_cryptography_list,
                    })

df = pd.DataFrame(dpr_result)
df.to_excel("./result/dpr/esr_dpr_result.xlsx", index=False)