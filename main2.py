import networkx as nx
import time
import pandas as pd
from methods import *
from  collections import Counter
import numpy as np
import sklearn
from sklearn.metrics import f1_score


start = time.time()

# open up the target pickle file
# graph = nx.read_gpickle("rug_mar.pkl")

# print(len(graph.nodes()))

# open up thenprocessed files
df_rug = pd.read_pickle("data//test_pickle.pkl")
df_wn = pd.read_pickle("data//smaller_corpus.pkl")


print('take a sample that is the length of the smallest')
#df_wn_all = df_wn.sample(len(df_rug))




print('divide into testing and training set')
#df_rug_train = df_rug.sample(frac=0.9, replace = True)
#df_wn_train = df_wn_all.sample(frac=0.9, replace = True)

#df_wn_test = df_wn_all[~df_wn_all.index.isin(df_wn_train.index)]
#df_rug_test = df_rug[~df_rug.index.isin(df_rug_train.index)]

doc_rug = list(df_rug.body)


from nltk.tokenize import RegexpTokenizer
token_pattern = r'^([a-zA-Z]+|\d+|\W)$'
tokenizer = RegexpTokenizer(token_pattern)

def justToken(token):
	token  = re.sub('[\W_]+', '', token)
	return token


		

doc_rug_train = sanitiseData(df_rug.body)
#doc_wn_train = sanitiseData(df_wn[df_wn.columns[0]])
checkList = []
for doc in doc_rug_train:
	checkList.extend(doc)

checkList = set(checkList)

df = pd.DataFrame({"checkList" : checkList})
df.to_csv("data//wordCheck.csv")	

'''
print(len(doc_rug_train))
print(len(doc_wn_train))

def removeLessThanTwo(docs):
    listee = []
    for doc in docs:
        if len(doc) > 2:
            listee.append(doc)

    return listee

doc_rug_train =   removeLessThanTwo(doc_rug_train)
doc_wn_train =   removeLessThanTwo(doc_wn_train)

print(len(doc_rug_train))
print(len(doc_wn_train))

# create labels
labs_wn =  list(np.array([0]*len(doc_wn_train)))
labs_rug =  list(np.array([1]*len(doc_rug_train)))

# create training corpus
training_corpus = doc_wn_train
training_labels = labs_wn

print(training_corpus)

pdf = pd.DataFrame({"corpus" : training_corpus})
pdf.to_csv("data//evaluation_data_eyeball.csv")

training_corpus.extend(doc_rug_train)
training_labels.extend(labs_rug)

print('size of training corpus')
print(len(training_corpus))
print(len(training_labels))

print()
print('clustering training_set')
df_min = 0.09
additional_data_points = []
for i in  range(0, 10):
	st = time.time()
	df_min = df_min - .01
	df, actual_labels, terms = clusterMethod(training_corpus, training_labels, df_min)
	f_score = f1_score(training_labels, actual_labels , average='binary')
	print("f_score >>> :")
	print(f_score)
	print(df_min)
	
	timet = (time.time() - st)/60
	additional_data_points.append([df_min, len(terms), f_score, timet])

df = pd.DataFrame({"metadata" : additional_data_points})
df.to_csv("data/metadata.csv")

'''
print(time.time() - start)
