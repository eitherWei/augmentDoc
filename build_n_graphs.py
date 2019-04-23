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
level = 3
timeArray = []
while level < 15:
    start = time.time()
    plotCorpusToDiGraph(corpusWN, "data//directedGraphs//direct_graph_wn_ply_" + str(level) + "_feb" ,  failSafe = False, level = level)
    plotCorpusToDiGraph(corpusRug, "data//directedGraphs//direct_graph_rugby_ply_" + str(level) + "_feb" ,  failSafe = False, level = level)
    print("we are at level: " + str(level))
    end = time.time()
    timeTaken = (end - start)/60
    print("timeTaken: " + str(timeTaken))
    timeArray.append(timeTaken)
    level = level + 1

print(timeArray)
print((time.time() - st)/60)
