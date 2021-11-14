# -*- coding: utf-8 -*-
import gensim.downloader 
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import gensim.models.fasttext as FT
from gensim.models.fasttext import FastText as FT_gensim
from pydantic import BaseModel
from typing import List

ft_en_cc = FT.load_facebook_vectors("models\cc.en.300.bin.gz")
nb_vec=  gensim.downloader.load('conceptnet-numberbatch-17-06-300') 
elib_model_s_100_e_5_w_5 = FT_gensim.load("models\elib_model_s_100_e_5_w_5")

model_dict = {'ft_en_cc': ft_en_cc, 'nb_vec':nb_vec, 'elib_model_s_100_e_5_w_5':elib_model_s_100_e_5_w_5}   

def get_model(model_name):
    """checks if model is available
    
    Args:
        model_name = name of model
        
    Returns:
        model: loaded Fasttext model from model_dict if found, False otherwise
    """
    if model_name in model_dict:
        return model_dict.get(model_name)
    else:
        return False
    
def get_similarity(model,word1,word2, nb_ft = False):
    """returns similarity of 2 words
    
    Args:
        model = fasttext model
        word1 = first word
        word2 = second word
        
    Returns:
        (float): similarity of given words
    """
    if model == nb_vec and not nb_ft:
        
        try: 
            return model.similarity('/c/en/'+word1.lower(),'/c/en/'+word2.lower())
        except:
            mod_w1 = word1
            while(len(mod_w1)>0):
                try: 
                    return model.similarity('/c/en/'+mod_w1.lower(),'/c/en/'+word2.lower())
                except: 
                    mod_w1 = mod_w1[:-1]
            
            if len(mod_w1) == 0:
                mod_w2 = word2
                while(len(mod_w2)>0):
                    try: 
                        return model.similarity('/c/en/'+word1.lower(),'/c/en/'+mod_w2.lower())
                    except: 
                        mod_w2 = mod_w2[:-1]
                        
            if len(mod_w2) ==0:
                mod_w1 = word1
                mod_w2 = word2
                while(len(mod_w1)>0 or len(mod_w2)>0):
                    try: 
                        return model.similarity('/c/en/'+mod_w1.lower(),'/c/en/'+mod_w2.lower())
                    except: 
                        if len(mod_w1)>0:   
                            mod_w1 = mod_w1[:-1]
                        if len(mod_w2)>0:
                            mod_w2= mod_w2[:-1]
            return 0
                
                       
    if model == nb_vec and nb_ft:  
        try: 
            return model.similarity('/c/en/'+word1.lower(),'/c/en/'+word2.lower())
        except:             
            return ft_en_cc.similarity(word1.lower(),word2.lower())
        
    else:
        return model.similarity(word1.lower(),word2.lower())
      
def get_vector(model,word):
    """returns vector of a word
    
    Args:
        model = fasttext model
        word = input word
        
    Returns:
        vector of given word
    """
    return model[word]

def get_most_similar(model,word,topn):
    """returns topn most similar words
    
    Args:
        model = fasttext model
        word = given word
        topn = number of most similar words required
        
    Returns:
        (list): similar words
    """
    return model.most_similar(positive=[word],topn=topn)



class Item(BaseModel):
    comparewords: List[str] = []
    
class Item_(BaseModel):
    words: List[str] = []

app=FastAPI()

@app.get("/")
def read_route():
    """root for the webservice
    
    Args:
        
    Returns:
    """
    return "hello world"

##New method
@app.post("/fasttext/{modelname}/similaritylist")
def read_similaritylist(modelname: str, baseword: str, item: Item):
    """returns similarity of 2 words if model is found, otherwise returns response 204
    
    Args:
        modelname = fasttext model name
        baseword = base word
        item = Item with comparewords (list of strings)
        
    Returns:
        (list): list of similarities of given words or response 204 if model not found
        
    """
   
    
        
    wordlist = item.comparewords
    nb_ft = False
    if modelname == 'nb_vec_ft':
        modelname = 'nb_vec'
        nb_ft = True

    model = get_model(modelname)
    if model == False:
        raise HTTPException(status_code=404, detail="Model not found", headers=None)
    else:
    
        similaritylist = [float(get_similarity(model, baseword, w, nb_ft)) for w in wordlist]
        return JSONResponse(content=similaritylist)

##New method
@app.post("/fasttext/{modelname}/wordvectors")
def read_vectorlist(modelname: str, item: Item_):
    """returns list of vector for given words if model is found, otherwise returns response 204
    
    Args:
        modelname = fasttext model name
        item = Item_ with words (list of strings)
        
    Returns:
        (list): list of vectors of given words or response 204 if model not found
        
    """
    wordlist = item.words
    model = get_model(modelname)
    if model == False:
        raise HTTPException(status_code=404, detail="Model not found", headers=None)
    else:
        vectorlist = []
        for w in wordlist:
            vec = get_vector(model, w)
            vec = vec.tolist()
            json_vec = jsonable_encoder(vec)
            vectorlist.append({'word':w, 'vector':json_vec})
        return JSONResponse(content=vectorlist)


@app.get("/fasttext/{modelname}/similarity")
def read_similarity(modelname: str, word1: str, word2: str):
    """returns similarity of 2 words if model is found, otherwise returns response 204
    
    Args:
        modelname = fasttext model name
        word1 = first word
        word2 = second word
        
    Returns:
        (float): similarity of given words or response 204 if model not found
        
    """
    model = get_model(modelname)
    if model == False:
        raise HTTPException(status_code=404, detail="Model not found", headers=None)
    else:
        val = get_similarity(model, word1, word2)
        return float(val)

@app.get("/fasttext/{modelname}/vector")
def read_vector(modelname: str, word: str):
    """returns vector of a word if model is found, otherwise returns response 204
    
    Args:
        modelname = fasttext model name
        word = input word
        
    Returns:
        (jsonlist): vector of given word or response 204 if model not found
        
    """
    model = get_model(modelname)
    if model == False:
        raise HTTPException(status_code=404, detail="Model not found", headers=None)
    else:
        vec = get_vector(model, word)
        vec = vec.tolist()
        json_vec = jsonable_encoder(vec)
        return JSONResponse(content=json_vec)

@app.get("/fasttext/{modelname}/most_similar")
def read_most_similar(modelname: str, word: str, topn: int):
    """returns topn most similar words if model is found, otherwise returns response 204
    
    Args:
        modelname = fasttext model name
        word = input word
        topn= number of required similar words
        
    Returns:
        (jsonlist): list of topn similar words or response 204 if model not found
        
    """
    model = get_model(modelname)
    if model == False:
        raise HTTPException(status_code=404, detail="Model not found", headers=None)
    else:
        word_list = get_most_similar(model, word, topn)
        json_list = jsonable_encoder(word_list)
        return JSONResponse(content=json_list)

    
