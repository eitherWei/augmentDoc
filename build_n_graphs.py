import pandas as pd
from preprocessMethods import *
from methods import *
import time
import os

st = time.time()
########################################################
# load up the required data
df_rug_feb = pd.read_pickle("data//rugby_feb.pkl")
df_rug_mar = pd.read_pickle("data//rug_mar.pkl")
df_wn = pd.read_pickle("data//world_news_mar.pkl")


# take the slice of wn that is required
df_wn_slice = df_wn[df_rug_mar.shape[0]: (df_rug_mar.shape[0] + 60000)]

# clean  the text
corpusWN = df_wn_slice.body.apply(cleanData)
corpusRug =  df_rug_feb.body.apply(cleanData)
level = 1
timeArray = []



# sanitiseData - removes puncutation , foreign terms , lowers, and stopword removal
df_rug_mar['sanitisedComment'] = df_rug_mar.body.apply(cleanData)
df_wn_slice['sanitisedComment'] = df_wn_slice.body.apply(cleanData)

print(len(df_rug_mar.sanitisedComment))
print(len(df_wn_slice.sanitisedComment))

print(df_rug_mar.columns)
def saveSanitisedFrame(listee, title):
    print("inside the method here is what it makes: ")
    df = pd.DataFrame({title : listee})


    #df.to_pickle("data//undirectedGraphs//direct_" + title + "augment")

    return df



while level < 15:
    df_result_terms = pd.DataFrame()
    start = time.time()
    altered_graph_rug = plotCorpusToDiGraph(list(corpusWN), "data//undirectedGraphs//undirect_graph_rug_ply_" + str(level) + "_feb" , graph = nx.Graph(), failSafe = False, level = level)
    altered_graph_wn = plotCorpusToDiGraph(list(corpusRug), "data//undirectedGraphs//undirect_graph_wn_ply_" + str(level) + "_feb" , graph = nx.Graph(),  failSafe = False, level = level)
    doc_rug_train , _ = augmentDocs(list(df_rug_mar.sanitisedComment), altered_graph_rug)
    doc_wn_train , _ = augmentDocs(list(df_wn_slice.sanitisedComment), altered_graph_wn)


    # sample of augmented documents
    print(len(doc_rug_train))
    print(len(doc_wn_train))


    print("we are at level: " + str(level))
    end = time.time()
    timeTaken = (end - start)/60
    print("timeTaken: " + str(timeTaken))
    timeArray.append(timeTaken)


    # create labels
    print("creating labels the length of the input")
    labs_rug =  list(np.array([1]*len(doc_rug_train)))
    print("len of rugby: " + str(len(labs_rug)))
    labs_wn =  list(np.array([0]*len(doc_wn_train)))
    print("len of wn: " + str(len(labs_wn)))
    labs_rug.extend(labs_wn)

    # create cluster list
    print("creating cluster list")
    cluster_list = []
    cluster_list.extend(doc_rug_train)
    cluster_list.extend(doc_wn_train)

    print(len(cluster_list))


    print("passing the vectors to the classifier for processing")
    terms, clusters = clusterMethod( cluster_list , labs_rug, 0.09)



    df = pd.DataFrame({"docs" + str(level) : cluster_list, "cat_pred": clusters.clusters, "cat_act" : labs_rug})


    df.to_pickle("data//undirectedGraphs//finalDf_"+ str(level))



    df_result_terms['level' + str(level)] = terms
    df_result_terms.to_pickle('data//undirectedGraphs//terms_' + str(level))
    level = level + 1
    print("level: " + str(level))



    # next we append the graph to the docs



print(timeArray)
print((time.time() - st)/60)
