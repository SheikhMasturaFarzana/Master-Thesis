import pandas as pd
import corpus_subset
import json

##########ENX##########
enx_corpus_dict = corpus_subset.get_corpus('ENX',1) 
biomedical_list = enx_corpus_dict['biomedical_list'] #96 
combustion_list = enx_corpus_dict ['combustion_list'] #25 
computer_list = enx_corpus_dict ['computer_list'] #138 
aerospace_list = enx_corpus_dict['aerospace_list'] #54 
chemical_list = enx_corpus_dict['chemical_list'] #82

dpr_data = []

[dpr_data.extend({'context': next(item for item in enx_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'biomedical'} for key in biomedical_list)]
[dpr_data.extend({'context': next(item for item in enx_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'combustion'} for key in combustion_list)]
[dpr_data.extend({'context': next(item for item in enx_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'chemical'} for key in chemical_list)]
[dpr_data.extend({'context': next(item for item in enx_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'computer'} for key in computer_list)]
[dpr_data.extend({'context': next(item for item in enx_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'aerospace'} for key in aerospace_list)]

with open(".\data_docs\enx_dpr_data.json", "w", encoding="utf-8") as f:
      json.dump(dpr_data, f, ensure_ascii=False)

##########ESR##########
      
esr_corpus_dict = corpus_subset.get_corpus('ESR',1)
deep_learning_list = esr_corpus_dict['deep_learning_list'] #37 
question_answering_list = esr_corpus_dict ['question_answering_list'] #35 
computer_vision_list = esr_corpus_dict ['computer_vision_list'] #34 
information_geometry_list = esr_corpus_dict['information_geometry_list'] #50 
cryptography_list = esr_corpus_dict['cryptography_list'] #38

dpr_data = []

[dpr_data.extend({'context': next(item for item in esr_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'deep learning'} for key in deep_learning_list)]
[dpr_data.extend({'context': next(item for item in esr_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'question answering'} for key in question_answering_list)]
[dpr_data.extend({'context': next(item for item in esr_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'cryptography'} for key in cryptography_list)]
[dpr_data.extend({'context': next(item for item in esr_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'Computer'} for key in computer_vision_list)]
[dpr_data.extend({'context': next(item for item in esr_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'agriculture'} for key in information_geometry_list)]

with open(".\data_docs\esr_dpr_data.json", "w", encoding="utf-8") as f:
      json.dump(dpr_data, f, ensure_ascii=False)
    

##########ENV##########
     
env_corpus_dict = corpus_subset.get_corpus('ENV',1)
energy_list = env_corpus_dict['energy_list'] #186
biodiversity_list = env_corpus_dict ['biodiversity_list'] #136
soil_list = env_corpus_dict ['soil_list'] #13
agriculture_list = env_corpus_dict['agriculture_list'] #41
chemicals_list = env_corpus_dict['chemicals_list'] #126

dpr_data = []

[dpr_data.extend({'context': next(item for item in env_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'energy'} for key in energy_list)]
[dpr_data.extend({'context': next(item for item in env_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'biodiversity'} for key in biodiversity_list)]
[dpr_data.extend({'context': next(item for item in env_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'chemicals'} for key in chemicals_list)]
[dpr_data.extend({'context': next(item for item in env_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'soil'} for key in soil_list)]
[dpr_data.extend({'context': next(item for item in env_corpus_dict['corpus'] if item['_key'] == key ).get('searchtext'), 'title': 'agriculture'} for key in agriculture_list)]

with open(".\data_docs\env_dpr_data.json", "w", encoding="utf-8") as f:
      json.dump(dpr_data, f, ensure_ascii=False)


