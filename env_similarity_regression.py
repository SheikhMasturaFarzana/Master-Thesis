import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score ,recall_score , f1_score, accuracy_score

result = []

# elib
combine_df = []
for filename in os.listdir("./result/document_retrieval/env_elib/"):

    df = pd.read_excel("./result/document_retrieval/env_elib/"+filename)
    df['label'] =(df.discipline == filename[9:-15]).map({True:1,False:0})
    combine_df.append(df)
    
    df= df.dropna()
    x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
    y = df.iloc[:,12]
    
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
    model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    
    temp = predicted.astype(int)
    y = Y_test[:, np.newaxis]
    y_p = temp[:, np.newaxis]
    
    result.append({'embedding':'elib',
                   'keyword':filename[9:-15],
                   'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                   'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                   'f1_score': "{:.2f}".format(f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
                   })

df = pd.concat(combine_df)
df= df.dropna()
x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
y = df.iloc[:,12]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
model.fit(X_train, Y_train)
predicted = model.predict(X_test)

temp = predicted.astype(int)
y = Y_test[:, np.newaxis]
y_p = temp[:, np.newaxis]

result.append({'embedding':'elib',
               'keyword':'combined',
               'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'f1_score': "{:.2f}".format(f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
               })



# ft
combine_df = []
for filename in os.listdir("./result/document_retrieval/env_ft/"):
    df = pd.read_excel("./result/document_retrieval/env_ft/"+filename)
    df['label'] =(df.discipline == filename[7:-15]).map({True:1,False:0})
    combine_df.append(df)
    
    df= df.dropna()
    x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
    y = df.iloc[:,12]
    
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
    model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    
    temp = predicted.astype(int)
    y = Y_test[:, np.newaxis]
    y_p = temp[:, np.newaxis]
    
    result.append({'embedding':'ft',
                   'keyword':filename[7:-15],
                   'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                   'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                   'f1_score': "{:.2f}".format(f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
                   })

df = pd.concat(combine_df)
df= df.dropna()
x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
y = df.iloc[:,12]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
model.fit(X_train, Y_train)
predicted = model.predict(X_test)

temp = predicted.astype(int)
y = Y_test[:, np.newaxis]
y_p = temp[:, np.newaxis]

result.append({'embedding':'ft',
               'keyword':'combined',
               'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'f1_score':"{:.2f}".format( f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
               })

# nb
combine_df = []
for filename in os.listdir("./result/document_retrieval/env_nb/"):
    df = pd.read_excel("./result/document_retrieval/env_nb/"+filename)
     
    df['label'] =(df.discipline == filename[7:-15]).map({True:1,False:0})
    combine_df.append(df)
    
    df= df.dropna()
    x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
    y = df.iloc[:,12]
    
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
    model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    
    temp = predicted.astype(int)
    y = Y_test[:, np.newaxis]
    y_p = temp[:, np.newaxis]
    
    result.append({'embedding':'nb',
                   'keyword':filename[7:-15],
                   'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                   'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                   'f1_score': "{:.2f}".format(f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
                   })

df = pd.concat(combine_df)
df= df.dropna()
x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
y = df.iloc[:,12]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
model.fit(X_train, Y_train)
predicted = model.predict(X_test)

temp = predicted.astype(int)
y = Y_test[:, np.newaxis]
y_p = temp[:, np.newaxis]

result.append({'embedding':'nb',
               'keyword':'combined',
               'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'f1_score':"{:.2f}".format( f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
               })

# nb_ft
combine_df = []
for filename in os.listdir("./result/document_retrieval/env_nb_ft/"):
    df = pd.read_excel("./result/document_retrieval/env_nb_ft/"+filename)
    
    df['label'] =(df.discipline == filename[10:-15]).map({True:1,False:0})
    combine_df.append(df)
    
    df= df.dropna()
    x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
    y = df.iloc[:,12]
    
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
    model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    
    temp = predicted.astype(int)
    y = Y_test[:, np.newaxis]
    y_p = temp[:, np.newaxis]
    
    result.append({'embedding':'nb_ft',
                   'keyword':filename[10:-15],
                   'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                   'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                   'f1_score': "{:.2f}".format(f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
                   })

df = pd.concat(combine_df)
df= df.dropna()
x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
y = df.iloc[:,12]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
model.fit(X_train, Y_train)
predicted = model.predict(X_test)

temp = predicted.astype(int)
y = Y_test[:, np.newaxis]
y_p = temp[:, np.newaxis]

result.append({'embedding':'nb_ft',
               'keyword':'combined',
               'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'f1_score':"{:.2f}".format( f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
               })

# ft_elib
combine_df = []
for filename in os.listdir("./result/document_retrieval/env_ft/"):
    df_1 = pd.read_excel("./result/document_retrieval/env_ft/"+filename)
     
    for f in os.listdir("./result/document_retrieval/env_elib/"):
        if (f[9:-15] == filename[7:-15]):
            

            df_2 = pd.read_excel("./result/document_retrieval/env_elib/"+f)

            df = pd.concat([df_1, df_2.drop(df_2.columns[[0, 1]], axis=1).add_prefix('elib_')], axis=1)
            df['label'] =(df.discipline == filename[7:-15]).map({True:1,False:0})
            
            combine_df.append(df)
            
            df= df.dropna()
            x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
            y = df.iloc[:,22]
            
            X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
            model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
            model.fit(X_train, Y_train)
            predicted = model.predict(X_test)
            
            temp = predicted.astype(int)
            y = Y_test[:, np.newaxis]
            y_p = temp[:, np.newaxis]
            
            result.append({'embedding':'ft_elib',
                           'keyword':filename[7:-15],
                           'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                           'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                           'f1_score': "{:.2f}".format(f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
                           })

df = pd.concat(combine_df)
df= df.dropna()
x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
y = df.iloc[:,22]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
model.fit(X_train, Y_train)
predicted = model.predict(X_test)

temp = predicted.astype(int)
y = Y_test[:, np.newaxis]
y_p = temp[:, np.newaxis]

result.append({'embedding':'ft_elib',
               'keyword':'combined',
               'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'f1_score':"{:.2f}".format( f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
               })


# ft_nb_og

combine_df = []
for filename in os.listdir("./result/document_retrieval/env_ft/"):
    df_1 = pd.read_excel("./result/document_retrieval/env_ft/"+filename)
     
    for f in os.listdir("./result/document_retrieval/env_nb/"):
        if (f[7:-15] == filename[7:-15]):
            
            df_2 = pd.read_excel("./result/document_retrieval/env_nb/"+f)

            df = pd.concat([df_1, df_2.drop(df_2.columns[[0, 1]], axis=1).add_prefix('nb_')], axis=1)
            df['label'] =(df.discipline == filename[7:-15]).map({True:1,False:0})
            
            combine_df.append(df)
            
            df= df.dropna()
            x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
            y = df.iloc[:,22]
            
            X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
            model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
            model.fit(X_train, Y_train)
            predicted = model.predict(X_test)
            
            temp = predicted.astype(int)
            y = Y_test[:, np.newaxis]
            y_p = temp[:, np.newaxis]
            
            result.append({'embedding':'ft_nb',
                           'keyword':filename[7:-15],
                           'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                           'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                           'f1_score': "{:.2f}".format(f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
                           })

df = pd.concat(combine_df)
df= df.dropna()
x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
y = df.iloc[:,22]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
model.fit(X_train, Y_train)
predicted = model.predict(X_test)

temp = predicted.astype(int)
y = Y_test[:, np.newaxis]
y_p = temp[:, np.newaxis]

result.append({'embedding':'ft_nb',
               'keyword':'combined',
               'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'f1_score':"{:.2f}".format( f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
               })

# ft_nb_ft

combine_df = []
for filename in os.listdir("./result/document_retrieval/env_ft/"):
    df_1 = pd.read_excel("./result/document_retrieval/env_ft/"+filename)
     
    for f in os.listdir("./result/document_retrieval/env_nb_ft/"):
        if (f[10:-15] == filename[7:-15]):
            
            
            df_2 = pd.read_excel("./result/document_retrieval/env_nb_ft/"+f)

            df = pd.concat([df_1, df_2.drop(df_2.columns[[0, 1]], axis=1).add_prefix('nb_ft_')], axis=1)
            df['label'] =(df.discipline == filename[7:-15]).map({True:1,False:0})
            
            combine_df.append(df)
            
            df= df.dropna()
            x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
            y = df.iloc[:,22]
            
            X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
            model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
            model.fit(X_train, Y_train)
            predicted = model.predict(X_test)
            
            temp = predicted.astype(int)
            y = Y_test[:, np.newaxis]
            y_p = temp[:, np.newaxis]
            
            result.append({'embedding':'ft_nb_ft',
                           'keyword':filename[7:-15],
                           'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                           'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                           'f1_score': "{:.2f}".format(f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
                           })

df = pd.concat(combine_df)
df= df.dropna()
x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
y = df.iloc[:,22]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
model.fit(X_train, Y_train)
predicted = model.predict(X_test)

temp = predicted.astype(int)
y = Y_test[:, np.newaxis]
y_p = temp[:, np.newaxis]

result.append({'embedding':'ft_nb_ft',
               'keyword':'combined',
               'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'f1_score':"{:.2f}".format( f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
               })


#ft_nb_elib

combine_df = []
for filename in os.listdir("./result/document_retrieval/env_ft/"):
    df_1 = pd.read_excel("./result/document_retrieval/env_ft/"+filename)
     
    for f in os.listdir("./result/document_retrieval/env_nb/"):
        if (f[7:-15] == filename[7:-15]):
            df_2 = pd.read_excel("./result/document_retrieval/env_nb/"+f)
            
            for d in os.listdir("./result/document_retrieval/env_elib/"):
                if (d[9:-15] == filename[7:-15]):
                    df_3 = pd.read_excel("./result/document_retrieval/env_elib/"+d)
            
                    df = pd.concat([df_1,
                                    df_2.drop(df_2.columns[[0, 1]], axis=1).add_prefix('nb_'),
                                    df_3.drop(df_3.columns[[0, 1]], axis=1).add_prefix('elib_')],
                                    axis=1)
                    
                    df['label'] =(df.discipline == filename[7:-15]).map({True:1,False:0})
                    
                    combine_df.append(df)
                    
                    df= df.dropna()
                    x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,
                                   12,13,14,15,16,17,18,19,20,21,
                                   22,23,24,25,26,27,28,29,30,31]]
                    y = df.iloc[:,32]
                    
                    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
                    model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
                    model.fit(X_train, Y_train)
                    predicted = model.predict(X_test)
                    
                    temp = predicted.astype(int)
                    y = Y_test[:, np.newaxis]
                    y_p = temp[:, np.newaxis]
                    
                    result.append({'embedding':'ft_nb_elib',
                                   'keyword':filename[7:-15],
                                   'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                                   'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
                                   'f1_score': "{:.2f}".format(f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
                                   })

df = pd.concat(combine_df)
df= df.dropna()
x = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,
              12,13,14,15,16,17,18,19,20,21,
              22,23,24,25,26,27,28,29,30,31]]
y = df.iloc[:,32]
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=40, stratify = y)
model = LogisticRegression(solver = 'liblinear',penalty='l2', C = 2e10) #to Avoid Regularization !!!
model.fit(X_train, Y_train)
predicted = model.predict(X_test)

temp = predicted.astype(int)
y = Y_test[:, np.newaxis]
y_p = temp[:, np.newaxis]

result.append({'embedding':'ft_nb_elib',
               'keyword':'combined',
               'precision': "{:.2f}".format(precision_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'recall' : "{:.2f}".format(recall_score(Y_test, predicted, pos_label = 1, average = 'binary')),
               'f1_score':"{:.2f}".format( f1_score(Y_test, predicted, pos_label = 1, average = 'binary'))
               })

result = sorted(result, key=lambda k: k['keyword']) 
df = pd.DataFrame(result)
df.to_excel("./result/document_retrieval/env_similarity_hist_eval.xlsx", index=False)  