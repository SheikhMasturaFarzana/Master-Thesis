"""
Calculates precision/recall on all documents in a directory
"""

import os
import pandas as pd
from operator import itemgetter

temp_eval_list = []
for filename in os.listdir("./result/dpr/"):

    df = pd.read_excel(str("./result/dpr/"+filename), dtype= {filename[:-16]+"_hit": int, "common_hit": int, "dpr_hits":int})
    query = df['keyword'].to_list()
    query = [b+1 for b,a in enumerate(query)]


    true_hit = df[filename[:-16]+"_hit"].to_list()
    common_hit = df['common_hit'].to_list()
    search_hits = df['dpr_hits'].to_list()
    
    p_num = 0
    p_denom = 0
    r_num=0
    r_denom=0
    temp = {}
    for d,e,c,z in zip(query, true_hit, common_hit, search_hits):
        if z !=0:
            p = c/z
        else:
            p = 0     
        r = c/e
        
        temp['q_'+str(d)+'_precision'] = "{:.2f}".format(p)
        temp['q_'+str(d)+'_recall'] = "{:.2f}".format(r)
        if (p+r) == 0:
            temp['q_'+str(d)+'_f1_score'] = 0
        
        else: 
            temp['q_'+str(d)+'_f1_score'] = "{:.2f}".format((2*p*r)/(p+r))
        
        
        if z !=0:
            p_num = p_num + ((c/z)*e)
        else:
            p_num = p_num
            
        p_denom = p_denom + e
        r_num = r_num +c
        r_denom = r_denom+e
       
    
    temp_eval_list.append(dict({'dataset': filename[:-16],
                                'w_precision':"{:.2f}".format(p_num/p_denom),
                                'w_recall':"{:.2f}".format(r_num/r_denom),
                                'w_f1_score': "{:.2f}".format((2*(p_num/p_denom)*(r_num/r_denom))/((p_num/p_denom)+(r_num/r_denom)))},
                                  **temp))
    


#temp_eval_list = sorted(temp_eval_list, key=itemgetter('w_precision'), reverse = True) 
df = pd.DataFrame(temp_eval_list)
#df.sort_values(['w_precision'], ascending=[True])
df.to_excel("./result/dpr/dpr_eval.xlsx", index=False)   

# esr_subset ['energy', 'biodiversity', 'soil', 'agriculture', 'chemicals']
# esr_subset ['biomedical', 'combustion', 'computer', 'aerospace', 'chemical']
# esr_subset['deep learning', 'question answering', 'computer vision', 'information geometry', 'cryptography']
   