from __future__ import print_function
import os
from methods_main import *
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram
import time
from collections import Counter
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
start = time.time()
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pickle
import numpy as np


###################################################
path = "Krapivin2009/all_docs_abstacts_refined"
doc_list, terms_list = extractDataFiles()
print(">>")
print(doc_list[0])

###################################################
# extract the text found in the files

docs = extractDocLines(path, doc_list)
#terms_goldStandard = extractDocLines(path, terms_list)
#df = pd.DataFrame({'doc' : docs})
#print(df.doc[0])

'''
count = 0

matches = []
pattern = '\((.*?)\)'
for doc in docs:
    for line in doc.split("."):
        if re.search(pattern, line):
            count = count + 1
            matches.append(line)


accroynms = extractAccronymns(matches)
print(len(accroynms.items()))
for k , v in accroynms.items():
    print(k , v)

df = pd.DataFrame({"Accronyms" : list(accroynms.keys()), "definitions" : list(accroynms.values())})

df.to_pickle("accroynm store")

print(df)
'''
###################################################
# process the data for clustering agent
df['sanitiseData'] = df.doc.apply(cleanData)
#df = pd.read_pickle("tfidf_min_04.pkl")
#print(df.head())
###################################################
# establish candidate terms by extracting most prevalent terms

# create/load a df_idf ranking of the corpus
tfidf_matrix, tfidf_vectoriser = applyTFidfToCorpus(df, failSafe = False)
print(type(tfidf_matrix))

# extract presaved tf_idf terms
df = ExtractSalientTerms(tfidf_vectoriser, tfidf_matrix, failSafe = False)

df_1 = df[df.doc_id_list == 2]

print(df_1.head())
print(df_1.shape)

#df_1.term_idf_list.sort(inplace = True)
df_1.sort_values(by=['term_idf_list'], inplace=True, ascending=False)
print(df_1.head(10))

print(terms_goldStandard[2])
#print(docs[0])



print("--------------")




print("----")
doc_id_list = []
term_list = []
term_idf_list = []



print(tfidf_matrix.shape[0])
print(tfidf_matrix.shape[1])

terms = tfidf_vectoriser.vocabulary_
keys = terms.keys()
values = terms.values()

dict1 = dict(zip(values, keys))



for i in range(0, (tfidf_matrix.shape[0])):
    for j in range(0, len(tfidf_matrix[i].indices)):
        doc_id_list.append(i)
        term_list.append(dict1[tfidf_matrix[i].indices[j]])
        term_idf_list.append(tfidf_matrix[i].data[j])

df = pd.DataFrame({"doc_id_list": doc_id_list, "term_list" : term_list, "term_idf_list": term_idf_list})
print(df.head())
print("----")
df.to_pickle("tfidf_whole_corpus.pkl")
'''
''''
df = pd.read_pickle("tfidf_whole_corpus.pkl")
print(df.head())

'''
'''
terms = tfidf_vectoriser.vocabulary_
print(terms)
keys = terms.keys()
values = terms.values()

dict1 = dict(zip(values, keys))
print(dict1)

#print(tfidf_matrix[0].indices)
print(tfidf_matrix[0].data)

for i in range(0, len(tfidf_matrix[0].indices)):
    print(tfidf_matrix[0].indices[i], dict1[i], tfidf_matrix[0].data[i])
###################################################

#print(type(tfidf_matrix[0]))
#print(tfidf_matrix[0].shape)




#x = tfidf_matrix[1].toarray()
#print(type(x))

#print(dir(tfidf_matrix[0]))

#print(tfidf_matrix[0].indices)
#print(tfidf_matrix[0].data)
'''
'''
def top_tfidf_feats(row, features, top_n=25):
    # Get top n tfidf values in row and return them with their corresponding feature names.
    topn_ids = np.argsort(row)[::-1]
    print(topn_ids)
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    # Top tfidf features in specific document (matrix row)
    row = np.squeeze(Xtr[row_id].toarray())
    #print(type(row))
    #print(type(Xtr[row_id].toarray()))
    return top_tfidf_feats(row, features, top_n)

df = top_feats_in_doc(tfidf_matrix, terms, 1)
'''
#print(df.head())

#df_terms = pd.DataFrame({"terms": list(terms.keys()), "index": list(terms.values())})

#save_tfidf = pd.DataFrame({"tfidf_matrix_04.pkl" : tfidf_matrix}, index=[0])
#save_tfidf.to_pickle("tfidf_min_04.pkl")
'''
#pickle.dump(tfidf_matrix, open("tfidf.pickle", "wb"))
#pickle.dump(train_comment_features, open("train_comment_features.pickle", "wb"))
#pickle.dump(test_comment_features, open("test_comment_features.pickle", "wb"))

'''
'''
###################################################
# extract the most frequent occuring terms
def get_n_top_words(corpus, n = None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x : x[1], reverse = True)

    return words_freq[:n]

terms = get_n_top_words(df.sanitiseData, n = 20)
top_df = pd.DataFrame(terms)
top_df.columns = ["Word", "Freq"]

# plot the terms
sns.set(rc={'figure.figsize':(13, 8)})
g = sns.barplot(x = "Word", y = "Freq", data = top_df)
g.set_xticklabels(g.get_xticklabels(), rotation = 30)
# format the terms for plotting
plt.show()
'''

print(10*"--*--")
print((time.time() - start)/60)
