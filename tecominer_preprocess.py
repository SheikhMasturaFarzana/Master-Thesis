# -*- coding: utf-8 -*-
import spacy
from nltk.stem.snowball import SnowballStemmer

from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
def custom_tokenizer(nlp):
    """ Custom spaCy tokenizer to keep hyphens (-) from getting tokenized.

    Args:
        nlp: spaCy lang model

    Returns:
        custom tokenizer

    """
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

spacy_en = {
    "lang_model": "en_core_web_sm",
    "max_nlp": 5000000
}

def is_mostly_digits(s):
    """ Checks whether at least half of the characters in the input string are digits.

    Args:
        s: Input string

    Returns:
        True or False depending on whether or not half of the characters in the input string are digits

    """
    return sum(c.isdigit() for c in s) >= len(s)/2


nlp = spacy.load(spacy_en['lang_model'])
nlp.max_length = spacy_en['max_nlp']
lang = spacy_en['lang_model'][:2]
stemmer = SnowballStemmer(language="english")
nlp.tokenizer  =custom_tokenizer(nlp)

def clean_text(text, pos, stop_words=None, short_len=1, stemming=False, exclude="", punctadd="."):
    """ Clean text, returning the cleaned text
    Cleaning means:
        - lemmatization or stemming
        - non-ascii strings removal
        - stop word removal
        - removal of words with unresolved pdf-cids
        - removal of short words
        - removal of mostly digits
        - removal of words with strange characters (optional)
        - removal of unwanted parts of speech
        - removal of repeated white space
        - removal of punctuation

    Args:
        text: string containing text to be cleaned
        pos: list of parts of speech to be kept in cleaned version
        stop_words (optional; default None results in model stop words): list of common words to be removed in any case
        short_len (optional; default 3): length of short words not to be considered for the cleaned corpus
        stemming (optional; default False): if True, stemming is used instead of lemmatization
        exclude (optional; default ""): string formed by characters which disqualify words for cleaned corpus
        punctadd (optional; defaul ""): string formed by characters which should function like punctuation marks
        
    Returns:
        (string) - cleaned text
    """
    
    if not stop_words:
        stop_words = nlp.Defaults.stop_words
    try:
        text_cleaned = ""
        for pc in punctadd:
            text = text.replace(pc, " ")
        text_nlp = nlp(text)
        
        if stemming:   
            tokens = [stemmer.stem(token.text.lower()) for token in text_nlp 
                if token.text.isascii()==True
                and token.text.lower() not in stop_words
                and "cid:" not in token.text
                and token.lemma_.lower() not in stop_words 
                and len(token.text)>short_len and not is_mostly_digits(token.text)
                and token.pos_ in pos and not token.is_punct | token.is_space
                and True not in [c in token.text for c in exclude]]
        else:
            tokens = [token.lemma_.replace(" ", "_").lower() for token in text_nlp 
                if token.text.isascii()==True
                and token.text.lower() not in stop_words
                and "cid:" not in token.text
                and token.lemma_.lower() not in stop_words 
                and len(token.text)>short_len and not is_mostly_digits(token.text)
                and token.pos_ in pos and not token.is_punct | token.is_space
                and True not in [c in token.text for c in exclude]]
        
        text_cleaned += " ".join(tokens)
                
    except:
        text_cleaned = ""
        
    return text_cleaned
