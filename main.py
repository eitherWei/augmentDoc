## inports
import pandas as pd
from methods import *
import time

start = time.time()

## read in the target to_pickle
# rugbyunionPickle
df1 = pd.read_pickle("test_pickle.pkl")
df = pd.read_pickle("world_news_mar.pkl")
#df_worldNews = pd.read_pickle("test_pickle.pkl")
print(df1.head())
print(df.head())

# extract a list of sentence vectors
vec = list(df1.body)
vec_wn = list(df.body)

#sanitise documents
corpus = sanitiseData(vec)
corpus_wn = sanitiseData(vec_wn)

## commented out for expediance plots the document lengths
#plotLenList(lenList)
#plotLenList(sanitised_lenList)

# create statistics on words after processing
#analysisCorpus(corpus)

G = plotCorpusToDiGraph(corpus, "rug_mar.pkl")
G_wn = plotCorpusToDiGraph(corpus_wn, "wn_mar.pkl")
#test_dict = dict(graph.edges.data())
#print(graph.edges.data())
# - error
tester = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse = True)
print(tester[:30])

tester = sorted(G_wn.edges(data=True), key=lambda x: x[2]['weight'], reverse = True)
print(tester[:30])



print(25*"-")
print("Time taken")
print((time.time() - start)/60)
print(25*"-")
