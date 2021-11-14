"""
Calculates precision/recall on all documents in a directory
"""

import pandas as pd
from operator import itemgetter

df = pd.read_excel("./result/document_retrieval/esr_similarity_hist_eval.xlsx", dtype= {'precision': float, "recall": float, "f1_score":float})

embedding = df['embedding'].unique()
#true_hit = {'biomedical':96, 'combustion':25, 'computer':138, 'aerospace': 54, 'chemical': 82, 'combined': 395}
#true_hit = {'energy':186, 'biodiversity':136, 'soil':13, 'agriculture':41, 'chemicals':126, 'combined': 502 }
true_hit = {'deep learning':37, 'question answering' : 35, 'computer vision':34, 'information geometry':50, 'cryptography':38, 'combined':194 }

temp_eval_list = []

for e in embedding:
    df_temp = df[df['embedding'] == e].to_dict('records')
    w_p = 0
    w_r = 0
    denom = 0
    for item in df_temp:
        w_p = w_p + (item['precision']*true_hit[item['keyword']])
        w_r = w_r + (item['recall']*true_hit[item['keyword']])
        denom = denom+true_hit[item['keyword']]
    
    w_p = w_p/denom
    w_r = w_r/denom
    w_f = (2*w_p*w_r)/(w_p+w_r) 
    temp_eval_list.append({'embedding': e, 'w_precision':  "{:.2f}".format(w_p), 'w_recall': "{:.2f}".format(w_r), 'w_f1-score':  "{:.2f}".format(w_f)})
    
temp_eval_list = sorted(temp_eval_list, key=itemgetter('w_precision'), reverse = True) 
df = pd.DataFrame(temp_eval_list)
df.to_excel("./result/document_retrieval/esr_similarity_hist_eval_weighted.xlsx", index=False)   

