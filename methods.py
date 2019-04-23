
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
from sklearn.metrics import f1_score , accuracy_score

# turning off the slicing warning
pd.set_option('mode.chained_assignment', None)

# removing unwanted symbols from document
token_pattern = r"(?u)\b\w\w+\b"
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


    tokenizer = RegexpTokenizer(token_pattern)
    doc  = tokenizer.tokenize(doc)

    return doc


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
    #lenList.sort()

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
            # chop array to facilitate recursion
            array = array[counter - 1:]

    return g , array

def plotCorpusToDiGraph(corpus, title , graph = nx.DiGraph(), failSafe = True, level = 2):
    ''' setting graph to default digraph unless otherwise stated '''
    # check if it is already created
    try:
        if (failSafe):
            ''' purposely crash try/except to force graph rebuild '''
            x = 1/0

        print("checking if graph exists ....")
        graph = nx.read_gpickle(title)
        print()

    except:
        print("FAILED to load graph")

        print("type of corpus")
        print(type(corpus))
        print("corpus length")
        print(len(corpus))
        print("Is the graph directed ? - " + str(nx.is_directed(graph)))
        for c in corpus:
            graph , _ = plotArray(c, level, graph)

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

def accuracyEval(clusters , labels):

    accuracy_score_1 = accuracy_score(labels , list(clusters.clusters), normalize=True)

    clusters = clusters.replace({0:1, 1:0})

    accuracy_score_2 = accuracy_score(labels , list(clusters.clusters), normalize=True)

    if accuracy_score_1 > accuracy_score_2:
        clusters = clusters.replace({0:1, 1:0})
        return accuracy_score_1 , clusters


    return accuracy_score_2 , clusters

def clusterMethod(cluster_list, labels, min_df):

    # initilise sklearn classifier
    vectoriser = TfidfVectorizer(max_df = 0.9, min_df = min_df,
                                                use_idf = False)
    # convert docs

    # have to stringify docs to satisfy sklearn format
    docs = stringifyText(cluster_list)
    matrix = vectoriser.fit_transform(docs)

    km = KMeans(n_clusters = 2)
    km.fit(matrix)
    clusters = km.labels_.tolist()

    clusters = pd.DataFrame({"clusters" : clusters, "actual_labels" : labels})

    pred = clusters.clusters.value_counts()

    print(pred)

    pred.sort_index(inplace = True)
    cols = list(pred.index)


    df = pd.DataFrame(index = cols, columns = cols)

    terms = vectoriser.get_feature_names()
    print("import terms: " + str(len(terms)))
    print(terms)



    for v in cols:
        f = clusters[(clusters.clusters == v)]
        act = f.actual_labels.value_counts()
        act.sort_index(inplace = True)
        df.iloc[v] = list(act)

    #df['actual_labels'] = list(f.actual_labels)
    df['pred total'] = df.sum(axis= 1)
    df.loc['actual total'] = df.sum(axis= 0).T

    acc , clusters = accuracyEval(clusters, labels)

    print("The accuracy for this approach: " + str(acc))
    print("return type extended to includes")

    return acc , terms , clusters


def extractAlternateGraph(pkl_title):
    print("opening: " + pkl_title)
    df_alter_thread = pd.read_pickle(pkl_title)
    print("shape: " + str(df_alter_thread.shape))
    print("sanitising " + pkl_title)
    doc_alter = sanitiseData(df_alter_thread.body)
    print("Graphing " + pkl_title)

    saveGraphTitle =   pkl_title
    G = plotCorpusToDiGraph(doc_alter, saveGraphTitle)
    return G

def createGraphDirectFromArray(df, pkl_title):
    doc_alter = sanitiseData(df.body)
    print("Graphing from direct list input")

    saveGraphTitle = "graph_" + pkl_title
    G = plotCorpusToDiGraph(doc_alter, saveGraphTitle)
    return G


def augmentTerm(term, graph ):
        # takes as argument term and finds graph compatriots
        lst = list(graph.edges(term, data = True))
        # orders list and returns top value
        lst.sort(key = lambda x: x[2]['weight'], reverse = True)
        #print(lst)
        # result is an ordered tuple set ((term, correlate, weight)...)
        # return top valued correlate
        if(len(lst) > 0):
            return [term, lst[0][1]]

        return ([term])

def augmentArray(array, graph):
        returnArray = []
        for a in array:
            if graph.has_node(a):
                term = augmentTerm(a, graph)
                returnArray.extend(term)
            else:
                #print("term missing from graph")
                returnArray.append(a)

        return returnArray


def augmentDocs(corpus, graph):
    print(" -- Augmenting Corpus -- ")
    augment_corpus = []
    #length_original = []
    #length_augmend = []
    for c in corpus:
        #length_original.append(len(c))
        augment_array = augmentArray(c, graph)
        augment_corpus.append(augment_array)
        #length_augmend.append(len(augment_array))

    df = pd.DataFrame()

    return augment_corpus , df


def read_raw_pickle():
    df = pd.read_pickle('poitics_mar.pkl')
    print(df.head())
    corpus = sanitiseData(df.body)
    print(len(corpus))

    df = pd.DataFrame({"corpus" : corpus})
    df.to_pickle("data//sanitised_politics.pkl")
