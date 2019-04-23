import nltk
import re
import numpy as np
from methods import *


def createMeasurements(corpus, len = 0):
    lengthList = []
    for c in corpus:
        if len(c) > len:
            lengthList.append(len(c))

    return lengthList

def measureArrayLength(corpus):
    return len(corpus)


def normaliseLength(listee, divisor):
    len = []
    for l in listee:
        v = float(l)/divisor
        len.append(int(v))
    return len

def removePoint(corpus, lent = 0):
    lengthList = []
    for c in corpus:
        if c < lent:
            lengthList.append(c)

    return lengthList

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

## replace author object with string
def swap(text):
    up = re.sub(r'.*_', '', text)
    return up

def tierExtract(df, id, level):

    t_1 = df['parent'].str.contains(id)

    return  t_1


def createContainsString(array):
    query = ""
    for a in array:
        query = query + a+ "|"

    #print(query[:-1])
    return query[:-1]

def extractThreadRefs(thread_records_df, level):
    index_list = list(thread_records_df.index)
    level_list =  list(np.array([level]*len(list(index_list))))
    dict1 = dict(zip(index_list, level_list ))

    return dict1

def createContainsString(array):
    query = ""
    for a in array:
        query = query + a+ "|"

    #print(query[:-1])
    return query[:-1]

def cleanData(text):
    # removes punctuation
    d = build_tokenizer(text)
    # cast to lowercase
    d = [x.lower() for x in d if x not in stop]

    return d
