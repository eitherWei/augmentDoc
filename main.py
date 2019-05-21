# work looks at documents found in Krapivin2009
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

start = time.time()
###################################################
# loop over the dataset and extract the text titles

path = "Krapivin2009/all_docs_abstacts_refined"
files = extractFileNames(path)
print("{} files found: ".format(len(files)))


suffix = ".txt"
doc_list = seperateBySuffix(files , suffix)
print("{} .txt files in directory".format(len(doc_list)))


suffix = ".key"
keyword_list = seperateBySuffix(files , suffix)
print("{} phrase files in directory".format(len(keyword_list)))

###################################################
# extract the text found in the files

docs = extractDocLines(path, doc_list)
print(len(docs))


vectorised_docs = []
for doc in docs:
    filtered_tokens = []
    for line in doc:
        if "--" not in line:
            tokens = tokeniseString(line)
            for token in tokens:
                # remove all non words
                if re.search('[a-zA-Z]', token):
                    filtered_tokens.append(token.lower())
    vectorised_docs.append(filtered_tokens)
print("lenght of processed file: " , len(vectorised_docs))

doc = vectorised_docs

###################################################
# extract the salient terms cast to tfidf




# initialise vectoriser
tfidf_vectoriser = TfidfVectorizer(max_df = 0.8, min_df = 0.2, stop_words ='english', tokenizer = tokenize_only)
tfidf_matrix = tfidf_vectoriser.fit_transform(docs)
print(tfidf_matrix.shape)
terms = tfidf_vectoriser.get_feature_names()
print(len(terms))


dist = 1 - cosine_similarity(tfidf_matrix)

###################################################
# cluster the docs
num_clusters = 5

km = KMeans(n_clusters = num_clusters)

km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

frame = pd.DataFrame({"clusters" : clusters})
print(frame.clusters.value_counts())

###################################################
# examining the clusters
print("Top terms per cluster")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("cluster %d words: " % i , end = '')

    for ind in order_centroids[i, :6]:
        print(terms[ind])
    print("\n\n")

###################################################
# convert distance matrix into a 2d matrix array

MDS()

mds = MDS(n_components = 2, dissimilarity = "precomputed", random_state = 1)
pos = mds.fit_transform(dist)

xs, ys = pos[:,0] , pos[:,1]


df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))

groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9))
ax.margins(0.05)
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            mec='none')

plt.show()

###################################################
# hierarchical clustering
linkage_matrix = ward(dist)

fig, ax = plt.subplots(figsize = (15, 20))
ax = dendrogram(linkage_matrix, orientation = "right")

plt.tight_layout()

plt.show()

###################################################


print(10*"--*---")
print((time.time() - start)/60)
