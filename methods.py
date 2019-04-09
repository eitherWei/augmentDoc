
#importd
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
import pandas as pd
import collections
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# removing unwanted symbols from document
token_pattern = '^([a-zA-Z]+|\d+|\W)$'
stop = set(stopwords.words('english'))

## tokenise word vectors
# method takes in string comment and returns a vector
def tokeniseString(v):
    tokens = nltk.word_tokenize(v)
    return tokens


def build_tokenizer(doc):
    "this is a function to a pruned sentence"
    from nltk.tokenize import RegexpTokenizer

    ''' remove url links , as they are noisy and dont add syntactical value '''
    doc = re.sub(r"http\S+", "", doc, flags=re.MULTILINE)
    doc = re.sub(r"www\S+", "", doc, flags=re.MULTILINE)


    #tokenizer = RegexpTokenizer(token_pattern)
    #doc  = tokenizer.tokenize(doc)
	
    doctored = []
    for token in doc.split(" "):
        token  = re.sub('[\W_]+', '', token)
        doctored.append(token)
    return doctored


def sanitiseData(liste):
    # method removes punctuation and casts to lower case.
    Docs = []
    for doc in liste:
        # removes punctuation
        d = build_tokenizer(doc)
        # cast to lowercase
        d = [x.lower() for x in d if x not in stop]
        Docs.append(d)

    return Docs

def plotLenList(lenList, title = None, xlabel = None, ylabel = "Density"):
    lenList.sort()

    #sns.distplot(lenList[10000: 60000])
    sns.distplot(lenList)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot()
    plt.show()

def analysisCorpus(corp):
    # create a giant array and apply Counter
    giantArray = []
    for c in corp:
        giantArray.extend(c)
    countedTerms = collections.Counter(giantArray)
    count = sorted(countedTerms.values())
    print(countedTerms.most_common(20))

    # remove terms that occur less than ten times
    count1 = [x for x in count if x > 50]
    plotLenList(count1, title = "Frequency Count Per Term", xlabel = "Term Frequency >50")

    print("The total number of words : " + str(len(giantArray)))

    # plotting vectors to the array
def plotArray(array, depth, g):
    counter = 1
    moveCounter = 1
    limit = len(array)

    for a in array:
        g.add_node(a)
        dummyDepth = depth
        # check that the point does not overrun the array
        if(counter + depth <= limit):
            # forage forward untill the maximum extent of the pointer is reached
            while(dummyDepth != 0):
                # check if weight already exists and update || create
                if g.has_edge(a, array[moveCounter]):
                    g[a][array[moveCounter]]['weight'] +=1
                else:
                    g.add_edge(a, array[moveCounter], weight= 1)
                # increment counters and reset depth

                dummyDepth = dummyDepth - 1
                moveCounter = moveCounter + 1

            counter = counter + 1
            moveCounter = counter
            dummyDepth = depth
        else:
            # chop array to vacilitate recursion
            array = array[counter - 1:]

    return g , array

def plotCorpusToDiGraph(corpus, title , failSafe = False):

    # check if it is already created
    try:
        if (failSafe):
            ''' purposely crash try/except to force graph rebuild '''
            x = 1/0

        print("checking if graph exists ....")
        graph = nx.read_gpickle(title)

    except:
        print("FAILED to load graph")
        graph = nx.DiGraph()
        for c in corpus:
            graph , _ = plotArray(c, 2, graph)

        nx.write_gpickle(graph, title)

    return graph

def createVocabCount(df):
    ''' method takes in a df of the data and returns a count '''
    corpus = sanitiseData(df.body)
    allterms = []
    for c in corpus:
        allterms.extend(c)

    countedTerms = collections.Counter(allterms)
    count = sorted(countedTerms.values())
    print(len(count))
    print(countedTerms['to'])

def stringifyText(augmentSents):

        stringArray = []

        for array in augmentSents:
            stringy = ""
            for word in array:
                stringy = stringy + " " + word
            stringArray.append(stringy)

        return stringArray

def clusterMethod(cluster_list, labels, min_df):

    # initilise sklearn classifier
    vectoriser = TfidfVectorizer(max_df = 0.9, min_df = min_df,
                                        use_idf = False)
    # convert docs

    # have to stringify docs to satisfy sklearn format
    docs = stringifyText(cluster_list)
    matrix = vectoriser.fit_transform(docs)
    terms = vectoriser.get_feature_names()
	
    print(terms)




    km = KMeans(n_clusters = 2)
    km.fit(matrix)
    clusters = km.labels_.tolist()

    print(len(clusters))
    print(len(labels))


    clusters = pd.DataFrame({"clusters" : clusters, "actual_labels" : labels})

    pred = clusters.clusters.value_counts()

    print(pred)

    pred.sort_index(inplace = True)
    cols = list(pred.index)


    df = pd.DataFrame(index = cols, columns = cols)


    for v in cols:
        f = clusters[(clusters.clusters == v)]
        act = f.actual_labels.value_counts()
        act.sort_index(inplace = True)
        df.iloc[v] = list(act)

    print('variables data')
    print(df.shape)
    print(len(f.actual_labels))
    #df['actual_labels'] = list(f.actual_labels)
    df['pred total'] = df.sum(axis= 1)
    df.loc['actual total'] = df.sum(axis= 0).T
    


    return df , km.labels_.tolist() , terms
