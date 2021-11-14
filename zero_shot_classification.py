def get_model(model_name):
    if model_name=='bart':
        from transformers import pipeline, BartForSequenceClassification, BartTokenizer
        return pipeline('zero-shot-classification', model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli'), tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli'))

    elif  model_name == "squeeze_bart":
        from transformers import pipeline, SqueezeBertTokenizer, SqueezeBertModel
        return pipeline('zero-shot-classification', model = SqueezeBertModel.from_pretrained('squeezebert/squeezebert-uncased'), tokenizer = SqueezeBertTokenizer.from_pretrained('squeezebert/squeezebert-uncased'))

    elif model_name == 'distil_bart':
        from transformers import pipeline, AutoModel, AutoTokenizer 
        return pipeline('zero-shot-classification', model = AutoModel.from_pretrained("valhalla/distilbart-mnli-12-3" ), tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-3" ))
    
    elif model_name == 'roberta':
        from transformers import pipeline, AutoModel, AutoTokenizer 
        return pipeline('zero-shot-classification', model = AutoModel.from_pretrained("roberta-large-mnli" ), tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli" ))

    elif model_name == 'deberta':
        from transformers import pipeline, DebertaTokenizer, DebertaForSequenceClassification
        return pipeline('zero-shot-classification', model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base'), tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base'))
    
    elif model_name == 'bart_yahoo':
        from transformers import pipeline, BartForSequenceClassification, BartTokenizer
        return pipeline('zero-shot-classification', model = BartForSequenceClassification.from_pretrained('joeddav/bart-large-mnli-yahoo-answers'), tokenizer = BartTokenizer.from_pretrained('joeddav/bart-large-mnli-yahoo-answers'))

    elif model_name == 'bert':
        from transformers import pipeline, BertForSequenceClassification, BertTokenizer
        return pipeline('zero-shot-classification', model = BertForSequenceClassification.from_pretrained("bert-base-uncased"), tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True))
    
    elif model_name == 'bert_enx':
        from transformers import pipeline, BertForSequenceClassification, BertTokenizer
        model = BertForSequenceClassification.from_pretrained("bert_enx")
        
        return pipeline('zero-shot-classification', model = model, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True))

    elif model_name == 'bert_esr':
        from transformers import pipeline, BertForSequenceClassification, BertTokenizer
        model =  BertForSequenceClassification.from_pretrained("bert_esr")    
        return pipeline('zero-shot-classification', model = model, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True))


def get_rank(model_name, corpus, searchtext, hypothesis, template):
    """ calculates and returns similarity scores between document and hypothesis.

    Args:
        lang_model: classifier name
        corpus: list of dict containing keys and searchtext
        searchtext: 'key' of searchtext field in corpus dict
        hypothesis: list of strings containing hypothesis
        template: hypothesis template, format "text {}."

    Returns:
        (list of dict): for each document list of scores and hypothesis pairs. 

    """
    output =[]
    classifier = get_model(model_name)
    for i,doc in enumerate(corpus):
        prediction = classifier(doc.get(searchtext), hypothesis, hypothesis_template = template, multi_label=True)
        scores = list(zip(prediction.get('labels'), prediction.get('scores')))
        output.append({'_key':doc.get('_key'), 'scores':scores})
        print(i)
    return output