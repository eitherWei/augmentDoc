import networkx as nx
import time
import pandas as pd
from methods import *
from  collections import Counter
import numpy as np
from sklearn.metrics import f1_score


title = "world_news_mar.pkl"
#title = "poitics_mar.pkl"
start = time.time()
# open up the target pickle file
#graph = nx.read_gpickle("rug_mar.pkl")

# print(len(graph.nodes()))

# open up the unprocessed files
df_rug_test = pd.read_pickle("test_pickle.pkl")
df_wn_all = pd.read_pickle("world_news_mar.pkl")

print('initial retrieved')
print(df_rug_test.shape)
print(df_wn_all.shape)

#df_wn = pd.read_pickle(title)

print('take a sample that is the length of the smallest')
df_wn_sample = df_wn_all.sample(len(df_rug_test))

print("length of testing corpus")
print(df_rug_test.shape)
print(df_wn_sample.shape)

#print(df_rug_train.head())
df_wn_build_graph = df_wn_all[~df_wn_all.index.isin(df_wn_sample.index)]
print('size of training graph for wn')
print(df_wn_build_graph.shape)

df_wn_build_graph = df_wn_build_graph.sample(len(df_rug_test), replace = True)

#print('divide into testing and training set')
#df_rug_train = df_rug.sample(frac=0.9, replace = True)
#df_wn_train_graph = df_wn_train.sample(len(df_rug_train.body), replace = True)

#print(df_rug_train.shape)
#print(df_wn_train_graph.head())
#print(df_wn_train_graph.shape)

print('length of training vector for wn')
print(df_wn_build_graph.shape)


# slight disparity on numbers , but not due to duplicates !?
print()
print('extracting text data')
print()


doc_rug_train = sanitiseData(df_rug_test.body)
doc_wn_train = sanitiseData(df_wn_sample.body)

print("testing doc set")
print(len(doc_rug_train))
print(len(doc_wn_train))

# generate wn graph proportional to rugby size
#df_wn_alternate = df_wn_all[~df_wn_all.index.isin(df_wn_train.index)]
#df_rug_test = df_rug[~df_rug.index.isin(df_rug_train.index)]
#df_wn_alternate = df_wn_all.sample(len(df_rug_train.body))



def removeLessThanTwo(docs):
    listee = []
    for doc in docs:
        if len(doc) > 2:
            listee.append(doc)

    return listee

doc_rug_train =   removeLessThanTwo(doc_rug_train)
doc_wn_train =   removeLessThanTwo(doc_wn_train)

# create labels
labs_wn =  list(np.array([0]*len(doc_wn_train)))
labs_rug =  list(np.array([1]*len(doc_rug_train)))



# create graph
## rugby
#alter_graph_rug = extractAlternateGraph("rugby_feb.pkl")
#doc_rug_train = augmentDocs(doc_rug_train, alter_graph_rug)
## wn


alter_graph_wn = createGraphDirectFromArray(df_wn_build_graph, title)
doc_wn_train , df = augmentDocs(doc_wn_train, alter_graph_wn)

print(df.head(40))

# create training corpus
training_corpus = doc_wn_train
training_labels = labs_wn


training_corpus.extend(doc_rug_train)
training_labels.extend(labs_rug)

print('size of training corpus')
print(len(training_corpus))
print(len(training_labels))

print()
print('clustering training_set')
df, actual_labels = clusterMethod(training_corpus, training_labels)



f_score = f1_score(training_labels, actual_labels , average='binary')
print("f_score >>> :")
print(f_score)


print(time.time() - start)
