from methods_main2 import *
from nlp_methods import *
import time
import re
from collections import OrderedDict

start = time.time()
path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
# extract all of the target directories

import nltk
#nltk.download('wordnet')

# class for preprocessing the text
methods = mainMethods(path)
#class for processing the text
nlp_m = nlpMethods()
individual_results_df = nlp_m.loadResultsDf("individual_results_df.pkl")

#path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/6/KEY/22.key"
#methods.extractKeyWordFilesTerms(1)
#extract the files from directory
files = methods.extractFileNames(path)
#initialise the methods dataframe with an identiier for each file
methods.df = pd.DataFrame({"handle" : files[:1]})
# extract the file names
methods.df['fileNames'] = methods.df.handle.apply(methods.extractFiles)
# extract the content from the files
methods.df['files'] = methods.df.fileNames.apply(methods.extractContent)


############################################
# extract the phrases present in the text stores in phraseDict of the class
df = methods.extractCorpusPhraseArray(methods.df['files'], failSafe = True)
'''
d  = df.phraseLists[0]

d = dict(sorted(d.items(), key=lambda x: x[1]))
############################################
print(dict(d))

# extract all of the accronyms from the corpus
#methods.extractAccronymns()

# process the data, removing puncutation stopwords and store as a string for processing
methods.df['sanitiseData'] = methods.df.files.apply(methods.cleanData)

#print("creating hypernym augmented text")
#methods.df['sanitiseDataHypernyms'] = methods.df.sanitiseData.apply(nlp_m.add_Hypernym)

#methods.df['lemmatisedWords'] = methods.df.sanitiseData.apply(nlp_m.lemmatise_corpus)


#methods.df['keywords'] = methods.df.handle.apply(methods.extractKeyWordFiles)
# list of singletons of impact terms , many folders are empty
methods.df['competition_terms'] = methods.df.handle.apply(methods.extractKeyWordFilesTerms)

# create/load a df_idf ranking of the corpus
tfidf_matrix, tfidf_vectoriser = methods.applyTFidfToCorpus(methods.df, failSafe = True)

# extract a df of all of the rated terms per document
df_terms = methods.ExtractSalientTerms(tfidf_vectoriser, tfidf_matrix, failSafe = True)

# order the df and assgin the class df with the top N terms per document
methods.extractTopNTerms(df_terms ,  N = 10)


result = nlp_m.evaluateTermResults(methods.df.competition_terms , methods.df.method_termDict)
print(type(result))
print(result)
r  = [x[0] for x in result]
r1 = [x[1] for x in result]
r2 = [x[2] for x in result]
print(sum(r))
print(sum(r1))
print(sum(r2))
individual_results_df['tfidf_lemmatised'] = result
individual_results_df.to_pickle("individual_results_df.pkl")

#print(df.head())

nlp_m.plotResults(individual_results_df)
'''

print(10*"-*-")
print((time.time() - start)/60)
