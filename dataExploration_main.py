import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
from methods import *
import time
from nltk.corpus import stopwords
from preprocessMethods import *
stop = set(stopwords.words('english'))
start = time.time()

'''
print("loading up the datasets")
# get the dfs rugby and wn
#df_rug_feb = pd.read_pickle("data//rugby_feb.pkl")
df_rug_mar = pd.read_pickle("data//rug_mar.pkl")
df_wn = pd.read_pickle("data//world_news_mar.pkl")
df_wn_slice = df_wn[:len(df_rug_mar)]




# sanitiseData - removes puncutation , foreign terms , lowers, and stopword removal
df_rug_mar['sanitisedComment'] = df_rug_mar.body.apply(cleanData)
df_wn_slice['sanitisedComment'] = df_wn_slice.body.apply(cleanData)


print("measuring the lengths of the sanitised arrays")
df_rug_mar['length']  = df_rug_mar.sanitisedComment.apply(measureArrayLength)
df_wn_slice['length']  = df_wn_slice.sanitisedComment.apply(measureArrayLength)


## pull out a suitable size from each
altered_graph_rug = pd.read_pickle("data//directedGraphs//direct_graph_rugby_ply_" + str(3) +"_feb")
doc_rug_train , _ = augmentDocs(list(df_rug_mar.sanitisedComment), altered_graph_rug)

altered_graph_wn = pd.read_pickle("data//directedGraphs//direct_graph_wn_ply_" + str(3) +"_feb")
doc_wn_train , _ = augmentDocs(list(df_wn_slice.sanitisedComment), altered_graph_wn)


df = pd.DataFrame({"rug3": doc_rug_train , "wn3" : doc_wn_train})
df.to_pickle("level3Dump.pkl")
'''
df = pd.read_pickle("level3Dump.pkl")
print(df.shape)

# removing all less than 3
df['length_rug'] = df.rug3.apply(measureArrayLength)
df['length_wn'] = df.wn3.apply(measureArrayLength)


print(df.head())
#df_rug_mar_greater_3 = df_rug_mar[df_rug_mar['length'] > 3]['augmentedComments']

cluster_list = []
rug_list = df[df['length_rug']> 3]['rug3']
wn_list = df[df['length_wn']> 3]['wn3']

cluster_list.extend(list(rug_list))
cluster_list.extend(list(wn_list))

# create a list of labels

rug_lab = list(np.array([1]*len(rug_list)))
wn_lab = list(np.array([0]*len(wn_list)))
labels = []
labels.extend(rug_lab)
labels.extend(wn_lab)
print(len(labels))


acc, terms, df = clusterMethod(cluster_list, labels, 0.1)
print(acc)
print(terms)
print(df.head())


print(" -- **************** --")
print((time.time() - start)/60)
