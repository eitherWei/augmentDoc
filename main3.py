import pandas as pd
import time
from preprocessMethods import *
from methods import *
import numpy as np
import networkx as nx
import os

st = time.time()

# get the dfs rugby and wn
df_rug_feb = pd.read_pickle("data//rugby_feb.pkl")
df_rug_mar = pd.read_pickle("data//rug_mar.pkl")
df_wn = pd.read_pickle("data//world_news_mar.pkl")


print("shape of the graphs")
print(df_rug_feb.shape)
print(df_rug_mar.shape)
print(df_wn.shape)
df_wn_slice = df_wn[:len(df_rug_mar)]
print(df_wn_slice.shape)
#print(df_rug_feb.shape)

# sanitiseData - removes puncutation , foreign terms , lowers, and stopword removal
df_rug_mar['sanitisedComment'] = df_rug_mar.body.apply(cleanData)
df_wn_slice['sanitisedComment'] = df_wn_slice.body.apply(cleanData)


#########################################################
print("part 2 expanding the clustering documents")
# first let us augment the rugby one
level  = 3
# presaved graph
#alter_graph_rug = pd.read_pickle("data//rugby_feb.pkl")
# reconstruct graph
#os.mkdir("data//undirectedGraphs//resultingDocs//")
corpus =  df_rug_feb.body.apply(cleanData)
df = pd.DataFrame()
result_df = pd.DataFrame()

index = list(range(3  , 15))
df_scores = pd.DataFrame(index = index, columns = ["import_terms", "accuracy"])
while level < 15:
    #altered_graph_rug = plotCorpusToDiGraph(corpus, "data//undirect_graph_rugby_ply2_feb" , graph = nx.Graph(),  failSafe = False)
    altered_graph_rug = pd.read_pickle("data//directedGraphs//direct_graph_rugby_ply_" + str(level) +"_feb")
    doc_rug_train , _ = augmentDocs(list(df_rug_mar.sanitisedComment), altered_graph_rug)

    altered_graph_wn = pd.read_pickle("data//directedGraphs//direct_graph_wn_ply_" + str(level) +"_feb")
    doc_wn_train , _ = augmentDocs(list(df_wn_slice.sanitisedComment), altered_graph_wn)

    # store the result for later analysis
    df["rug" + str(level)] = doc_rug_train
    df["wn" + str(level)] = doc_wn_train

    # append the result to the dataframe
    df_rug_mar['augmentedComments'] = doc_rug_train
    df_wn_slice['augmentedComments'] = doc_wn_train

    print("measuring the lengths of the sanitised arrays")
    df_rug_mar['length']  = df_rug_mar.augmentedComments.apply(measureArrayLength)
    df_wn_slice['length']  = df_wn_slice.augmentedComments.apply(measureArrayLength)

    print("taking a slice greater than 3")
    df_rug_mar_greater_3 = df_rug_mar[df_rug_mar['length'] > 3]['augmentedComments']
    df_wn_slice_greater_3 = df_wn_slice[df_wn_slice['length'] > 3]['augmentedComments']

    print("creating the test set")
    cluster_list = []
    cluster_list.extend(list(df_rug_mar_greater_3))
    cluster_list.extend(list(df_wn_slice_greater_3))

    # create labels
    print("creating labels the length of the input")

    labs_rug =  list(np.array([1]*df_rug_mar_greater_3.shape[0]))
    print("len of rugby: " + str(len(labs_rug)))
    labs_wn =  list(np.array([0]*df_wn_slice_greater_3.shape[0]))
    print("len of wn: " + str(len(labs_wn)))
    labs_rug.extend(labs_wn)

    labels = labs_rug
    print("length of the two arrays: " + str(len(labels)))

    print("passing the vectors to the classifier for processing")
    acc, terms = clusterMethod( cluster_list , labels, 0.1)


    df_scores.loc[level]["import_terms"] = terms
    df_scores.loc[level]["accuracy"] = acc

    print(df_scores)

    level = level + 1
    print("level is :" + str(level))

    df.to_csv("_dump_data.csv")
    df_scores.to_csv("_dump_stats.csv")


df.to_pickle("data//directedGraphs//resultingDocs//augmentedCorpora_dataframe")

df_scores.to_pickle("data//directedGraphs//resultingDocs//results_n_terms")


print(df_scores)

#print("we will take from the last point in the test set, to an equal amount of comments to the rugby set")
#df_worldNews_graph_docs = df_wn[len(df_rug_mar): len(df_rug_mar) + 60000]

#print("size of alterniative corpus")
#print(df_worldNews_graph_docs.shape)
#corpus =  df_worldNews_graph_docs.body.apply(cleanData)
# creating a new graph
#altered_graph_wn = plotCorpusToDiGraph(corpus, "data//undirect_graph_wn_ply2_60000" ,graph = nx.Graph(), failSafe = False)
#doc_wn_train , _ = augmentDocs(list(df_wn_slice.sanitisedComment), altered_graph_wn)

#print(df_rug_mar.shape)
#df_rug_mar['sanitisedComment'] = doc_rug_train
#df_wn_slice['sanitisedComment'] = doc_wn_train

#########################################################



#print("measuring the lengths of the sanitised arrays")
#df_rug_mar['length']  = df_rug_mar.sanitisedComment.apply(measureArrayLength)
#df_wn_slice['length']  = df_wn_slice.sanitisedComment.apply(measureArrayLength)

#print("taking a slice greater than 3")
#df_rug_mar_greater_3 = df_rug_mar[df_rug_mar['length'] > 3]['sanitisedComment']
#df_wn_slice_greater_3 = df_wn_slice[df_wn_slice['length'] > 3]['sanitisedComment']


#print("creating the test set")
#cluster_list = []
#cluster_list.extend(list(df_rug_mar_greater_3))
#cluster_list.extend(list(df_wn_slice_greater_3))

# create labels
#print("creating labels the length of the input")
#print(df_rug_mar.shape[0])

#labs_rug =  list(np.array([1]*df_rug_mar_greater_3.shape[0]))
#print("len of rugby: " + str(len(labs_rug)))
#labs_wn =  list(np.array([0]*df_wn_slice_greater_3.shape[0]))
#print("len of wn: " + str(len(labs_wn)))
#labs_rug.extend(labs_wn)

# pass the labels to a neutral tag
#labels = labs_rug
#print("length of the two arrays: " + str(len(labels)))

#print("the length of labels are: ")
#print(len(cluster_list))
#print(len(labels))


#print("passing the vectors to the classifier for processing")
#min_df = 0.1
#accList = []
#termLengthList = []

#while min_df > 0.01:
#    print(min_df)
#    acc, termLength = clusterMethod( cluster_list , labels, min_df)
    #accList.append(acc)
    #termLengthList.append(termLength)
    #min_df = min_df - 0.01

#print(accList)
#print(termLengthList)

print(10*"-*-")
print(time.time() - st)
