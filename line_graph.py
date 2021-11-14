import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os

######## Document Retreival ########
for filename in os.listdir("./result/relevance_ranking/precision_recall_eval/"):

    df = pd.read_excel("./result/relevance_ranking/precision_recall_eval/"+filename)
    
    x = df['st_rt']
    plt.figure(figsize=(8,5))
    ax = plt.gca()
    
    df.plot(kind='line',x='st_rt',y='w_precision',color='blue',ax=ax)
    df.plot(kind='line',x='st_rt',y='w_recall', color='red', ax=ax)
    df.plot(kind='line',x='st_rt',y='w_f1_score', color='green', ax=ax)
    

    plt.xticks(range(len(x)), x[::1], rotation=75)
    #plt.xticks(range(len(x)), x)

    plt.grid()
    
    plt.savefig('./result/relevance_ranking/eval_graphs/'+filename[:-5]+'.png',bbox_inches = 'tight')
    plt.show()

#%%
######## Relevance Ranking ########

#ENX

for filename in os.listdir("./result/document_retrieval/"):
    if filename.endswith('.xlsx'):
        df = pd.read_excel("./result/document_retrieval/"+filename)
        keywords = df['keyword'].unique()
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan']
        metrics =  ['precision', 'recall', 'f1_score']
        
        for metric in metrics:
            plt.figure(figsize=(8,5))
            ax = plt.gca()
            for color, keyword in zip(colors,keywords):
                df_1 = df[df['keyword'] == keyword]
                df_1.plot(kind='line',x='embedding',y=metric,color=color,label = keyword, ax=ax)
            plt.grid()
            plt.savefig('./result/document_retrieval/eval_graphs/'+filename[:-5]+'_'+metric+'.png')
            plt.show()     
            
#%%
df = pd.read_excel("./result/document_retrieval/esr_similarity_hist_eval_weighted.xlsx")
    
x = df['embedding']
plt.figure(figsize=(8,5))
ax = plt.gca()

df.plot(kind='line',x='embedding',y='w_precision',color='blue',ax=ax)
df.plot(kind='line',x='embedding',y='w_recall', color='red', ax=ax)
df.plot(kind='line',x='embedding',y='w_f1-score', color='green', ax=ax)


plt.xticks(range(len(x)), x[::1])
#plt.xticks(range(len(x)), x)

plt.grid()

plt.savefig('./result/document_retrieval/eval_graphs/esr_weighted.png',bbox_inches = 'tight')
plt.show()