import os
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

def extractFileNames(path):
    fileNames = []
    for r, d, f in os.walk(path):
        for file in f:
            fileNames.append(file)

    return fileNames

def seperateBySuffix(file_list , suffix):
    #loop over filelist and identify target docs
    target_list = []
    for filename in file_list:
        if  suffix in filename:

            target_list.append(filename)

    target_list.sort()
    print(target_list[0])
    return target_list

def extractDocLines(path, doc_list):
    docs_array = []
    for d in doc_list:
        doc_addr = path + "/" + d
        doc = open(doc_addr, "rb")
        lines_array = []
        document = ""
        for line in doc.readlines():
            document = document + line

        docs_array.append(document)

    return docs_array

def tokeniseString(v):
    filtered_tokens = []
    tokens = nltk.word_tokenize(v)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def cleanData(data):

    data = re.sub('[^a-zA-Z-]', ' ', data)
    data = re.sub("--", "", data)
    # remove punctuation + lower case
    data = data.split()
    data = [x.lower() for x in data if x not in stop]
    data = [x for x in data if len(x) > 1]

    data = " ".join(data)

    return data

def extractDataFiles():
    # loop over the dataset and extract the text titles

    path = "Krapivin2009/all_docs_abstacts_refined"
    files = extractFileNames(path)
    print("{} files found: ".format(len(files)))


    suffix = ".txt"
    doc_list = seperateBySuffix(files , suffix)
    print("{} .txt files in directory".format(len(doc_list)))


    suffix = ".key"
    keyword_list = seperateBySuffix(files , suffix)
    print("{} .key files in directory".format(len(keyword_list)))

    return doc_list , keyword_list


def applyTFidfToCorpus(df, failSafe = False):
    # create tf-idf matrix for the corpus
    tfidf_matrix = None
    try:
        if (failSafe):
            ''' purposely crash try/except to force vectoriser rebuild '''
            x = 1/0

        print("-- Retrieving stored tfidf_matrix --")

        tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb" ) )
        tfidf_vectoriser = pickle.load(open("tfidf_vectoriser.pkl", "rb" ) )


    except:

        print("failed to load -- building tokeniser --")
        # initialise vectoriser and pass cleaned data
        tfidf_vectoriser = TfidfVectorizer(max_df = 0.8, min_df = 0.4, stop_words ='english', tokenizer = tokenize_only)
        tfidf_matrix = tfidf_vectoriser.fit_transform(list(df.sanitiseData))

        #df= pd.DataFrame({"tfidf_matrix" : tfidf_matrix}, index=[0])
        #save_tfidf.to_pickle("tfidf_min_04.pkl")
        #df.to_pickle("tfidf_matrix.pkl")

        # pickle tfidf matrix for faster future load
        with open("tfidf_matrix.pkl", 'wb') as handle:
                    pickle.dump(tfidf_matrix, handle)

        # pickle tfidf vectoriser for faster future load
        with open("tfidf_vectoriser.pkl", 'wb') as handle:
                    pickle.dump(tfidf_vectoriser, handle)

    return tfidf_matrix , tfidf_vectoriser

def ExtractSalientTerms(tfidf_vectoriser, tfidf_matrix, failSafe = False):
    df = pd.DataFrame()
    try:
        if (failSafe):
            ''' purposely crash try/except to force vectoriser rebuild '''
            x = 1/0

        print("loading presaved processed corpus --")
        df = pd.read_pickle("tfidf_whole_corpus.pkl")
        # lists for storing data

    except:
        print(" failed to load terms -- rebuilding -- ")
        doc_id_list = []
        term_list = []
        term_idf_list = []

        # extract terms from vectoriser
        terms = tfidf_vectoriser.vocabulary_
        keys = terms.keys()
        values = terms.values()

        # invert the dict so the keys are the values and values the keys
        dict1 = dict(zip(values, keys))

        # iterate through matrix
        for i in range(0, (tfidf_matrix.shape[0])):
            for j in range(0, len(tfidf_matrix[i].indices)):
                # append the appropriate list with the appropriate value
                doc_id_list.append(i)
                term_list.append(dict1[tfidf_matrix[i].indices[j]])
                term_idf_list.append(tfidf_matrix[i].data[j])

        # cast to dataframe
        df = pd.DataFrame({"doc_id_list": doc_id_list, "term_list" : term_list, "term_idf_list": term_idf_list})
        # pickle process for future fast retrieval
        df.to_pickle("tfidf_whole_corpus.pkl")

    return df

def extractCandidateSubstring(match):
    pattern = '\((.*?)\)'
    candidate = ""
    substring = ""
    match = match.strip("\n")
    match = match.split(" ")
    for i in range(0, len(match)):
        cand = re.search(pattern, match[i])
        if cand:
            candidate = cand.group(1)
            # check that it is longer than 1
            if len(candidate) > 1:
                # check and remove for non capital mix
                if(lookingAtAcroynms(candidate)):
                    candidate = removeNonCapitals(candidate)
                j = len(candidate)
                substring = match[i-j:i]
                # check if accronym is present
                wordsAccro = returnPotentAccro(substring)
                if candidate.lower() == wordsAccro.lower():
                    # return the correct accro and definition
                    return (candidate, substring)

    return(None, None)

# check of the main lettes match
def returnPotentAccro(substring):
    firsts = ""
    for s in substring:
        if(len(s) > 0):
            firsts = firsts + s[0]
    return firsts


def lookingAtAcroynms(accro):
    # case one check if accroynm has an append s
    bool = False
    for s in accro[:1]:
        if s.isupper:
            bool = True

    return bool


def removeNonCapitals(accro):
    string = ""
    for s in accro:
        if s.isupper():
            string = string + s
    return string



def extractAccronymns(matches):
    accroDict = {}
    for match in matches:
        acc , substring = extractCandidateSubstring(match)
        if(acc not None):
            if len(acc) != 1:
                accroDict[acc] = substring


    return accroDict
