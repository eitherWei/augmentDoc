import pandas as pd
from methods_main2 import *
from methods_main3 import *
import matplotlib.pyplot as plt
import re
# a place for all things and a thing for all places
import time

start = time.time()

crashIt = False
methods1 = path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
otherPath = "Krapivin2009/all_docs_abstacts_refined"
dataset = pd.DataFrame()

methods1 = processMethods()
methods = mainMethods(path)

methods.extractFileNames(path)

try:
    methods.df = pd.read_pickle("saveTimeFrame.pkl")

except:
    methods.df['fileNames'] = methods.df.handle.apply(methods.extractFiles)

    methods.df['files'] = methods.df.fileNames.apply(methods.extractContent)

    methods.df['sanitiseData'] = methods.df.files.apply(methods1.cleanData)

    #tfidf_matrix, tfidf_vectoriser = methods.applyTFidfToCorpus(methods.df.sanitiseData, failSafe = crashIt)

    #methods.tfidf_matrix = tfidf_matrix
    #methods.tfidf_vectoriser = tfidf_vectoriser

    methods.df['keywords'] = methods.df.handle.apply(methods.extractKeyWordFiles)
    methods.df['competition_terms'] = methods.df.handle.apply(methods.extractKeyWordFilesTerms)

    methods.df.keywords = methods.df.keywords.apply(methods.lemmatiseCompTerms)

    methods.df.to_pickle("saveTimeFrame.pkl")

termsArray = []
for terms in methods.df.keywords:
    termsArray.extend(terms)
for terms in methods.df.competition_terms:
    if not isinstance(terms, int):
        termsArray.extend(terms)
#print(methods.df.head())
print(len(termsArray))
termsDict = dict(Counter(termsArray))
print(len(termsDict))

#######################################################################
print("============================")

dataset = pd.DataFrame()


methods.df['ref'] = methods.df.files.apply(methods.extractRefernces)


#print(methods.df.ref.value_counts())


# process references
def cleanReferences(text):
    if text is not None:
        text = text.split(". ")
        cleanedText = []
        for sent in text:
            sent = methods1.lemmatise_corpus(sent)
            sent = re.sub('[^a-zA-Z]', ' ' , sent)
            cleanedText.append(sent)
        return cleanedText

methods.df['cleanedRefs'] = methods.df.ref.apply(cleanReferences)
print("--- gappa gappa ------")
# methods somewhat defunct as the wordlist is long and the looking woudl mean string by string
# extract the approapriate words from the appraopriate documents


#######################################################################
# extracting target terms
count = 0
def appendKeyTerms(df):
    list = []
    for i in range(df.shape[0]):
        if not isinstance(df.competition_terms[i], int):
            list.append(df.keywords[i] + df.competition_terms[i])
        else:
            list.append(df.keywords[i])
    return list


methods.df['allCompTerms'] = appendKeyTerms(methods.df)
#######################################################################
count = 0
miss = 0
all_list = []
for i in range(methods.df.shape[0]):
    if methods.df.allCompTerms[i] == None:
        all_list.append(None)
    else:
        rowList = []
        for term in methods.df.allCompTerms[i]:
            if re.findall(term, methods.df.sanitiseData[i]):
                rowList.append(term)
                count = count + 1
            else:
                miss = miss + 1
        all_list.append(rowList)

methods.df['presentTerms'] = all_list
#######################################################################
# playing around with extract chunks
def extractSection(df):
    print(df.files[0])


extractSection(methods.df)
for i in  list(methods.df.allCompTerms):
    print()
    print(i)
    print()
    print(10*"-><-")

methods.df.to_pickle("compTermInstance.pkl")
#######################################################################

'''
for line in methods.df.cleanedRefs[1]:
    print(line)
    print(10*"-x-")
'''
# no internet --> to be used when processing the strings
'''
sample = '(LDA)'
pattern = '\((.*?)\)'
cand = re.search(pattern, sample)
print(cand)
'''

#######################################################################
print(10*"-*-")
print((time.time()  - start)/60)
