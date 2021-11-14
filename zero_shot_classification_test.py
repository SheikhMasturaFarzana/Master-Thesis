import pandas as pd
import corpus_subset
#import zero_shot_classification as model

env_corpus_dict = corpus_subset.get_corpus('ENV',1)
energy_list = env_corpus_dict['energy_list']
biodiversity_list = env_corpus_dict ['biodiversity_list']
soil_list = env_corpus_dict ['soil_list']
agriculture_list = env_corpus_dict['agriculture_list']
chemicals_list = env_corpus_dict['chemicals_list']

#possible model_name = 'bart', 'squeeze_bart', 'distil_bart', 'roberta', 'deberta', 'bart_yahoo', 'bert', 'bert_enx', 'bert_esr'
#template format= "text {}." 'This is a document about',
model_names = ['bart','bert_esr']
templates = ['','This document is about', 'This example is', 'This is an example of', 'This is a document about', 'This is a text about', 'This text contains', 'This text is about',  'This topic is']

for model_name in model_names:
    for template in templates:
        import zero_shot_classification as model
#model_name = 'bart'
#template = 'This is a document about'

        env_scores = model.get_rank(model_name = model_name,
                                    corpus = env_corpus_dict['corpus'],
                                    searchtext = 'searchtext',
                                    hypothesis = ['energy','agriculture', 'biodiversity', 'soil', 'chemicals'],
                                    template = template+' {}.')


        queries = [item[0] for item in env_scores[0].get('scores')]
        queries_dict = {key: [] for key in queries}
       
    
        for doc in env_scores:
            queries_dict[doc.get('scores')[0][0]].append(doc.get('_key')) 
            

        env_subset = []        
        env_subset.append({'keyword': 'energy' ,
                            'env_hit': len(energy_list),
                            'env_list': energy_list,
                            'common_hit':len(list(set(energy_list)&set(queries_dict.get('energy')))),
                            'common_list': list(set(energy_list)&set(queries_dict.get('energy'))), 
                            'zero_shot_hits': len(queries_dict.get('energy')),
                            'zero_shot_list': queries_dict.get('energy'),
                            })
        
        env_subset.append({'keyword': 'biodiversity' ,
                            'env_hit': len(biodiversity_list),
                            'env_list': biodiversity_list,
                            'common_hit':len(list(set(biodiversity_list)&set(queries_dict.get('biodiversity')))),
                            'common_list': list(set(biodiversity_list)&set(queries_dict.get('biodiversity'))), 
                            'zero_shot_hits': len(queries_dict.get('biodiversity')),
                            'zero_shot_list': queries_dict.get('biodiversity'),
                            
                            })
        env_subset.append({'keyword': 'soil' ,
                            'env_hit': len(soil_list),
                            'env_list': soil_list,
                            'common_hit':len(list(set(soil_list)&set(queries_dict.get('soil')))),
                            'common_list': list(set(soil_list)&set(queries_dict.get('soil'))), 
                            'zero_shot_hits': len(queries_dict.get('soil')),
                            'zero_shot_list': queries_dict.get('soil'),
                            
                            })
        env_subset.append({'keyword': 'agriculture' ,
                            'env_hit': len(agriculture_list),
                            'env_list': agriculture_list,
                            'common_hit':len(list(set(agriculture_list)&set(queries_dict.get('agriculture')))),
                            'common_list': list(set(agriculture_list)&set(queries_dict.get('agriculture'))), 
                            'zero_shot_hits': len(queries_dict.get('agriculture')),
                            'zero_shot_list': queries_dict.get('agriculture'),
                            
                            })
        env_subset.append({'keyword': 'chemicals' ,
                            'env_hit': len(chemicals_list),
                            'env_list': chemicals_list,
                            'common_hit':len(list(set(chemicals_list)&set(queries_dict.get('chemicals')))),
                            'common_list': list(set(chemicals_list)&set(queries_dict.get('chemicals'))), 
                            'zero_shot_hits': len(queries_dict.get('chemicals')),
                            'zero_shot_list': queries_dict.get('chemicals'),      
                            
                            })
        
        df = pd.DataFrame(env_subset)
        df.to_excel("./result/template_eval/env_zero_shot_"+model_name+"_"+'_'.join(template.split())+".xlsx", index=False)
        print('done')