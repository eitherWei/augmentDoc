
from getData import returnData
import pandas as pd
import re
import numpy as np

import networkx as nx

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

stop = set(stopwords.words('english'))

token_pattern = r"(?u)\b\w\w+\b"

vectorLength = 2


class processingClass():

    # initialise the class variables
    def __init__(self, name, cut, dataSlice = 100000):
        # data source for processing
        self.name = name
        # matrix of all the vectorised data created below
        self.dataList = []
        # variable for managing size of input data
        self.dataSlice = dataSlice
        # variable to hold raw text preprocessing
        self.df = pd.DataFrame()
        # unpack class data
        self.returnPickle()
        # automatically process input data
        #self.vectoriseData()
        # class graph containing all of the semantic metrics
        self.graph = nx.Graph()
        # test set populated in returnTestData method
        self.testSet = []
        # final augment testSet
        self.augmentSents = []
        # division for testSet
        self.slice = 0
        # cut off point for testData
        self.cut = cut


    def add_synonymns(self):
        A_testSet = []
        for array in self.testSet:
            A_testSet.append(self.add_syn_sent(array))

        self.augmentSents = A_testSet

    def add_hypernyms(self):
        A_testSet = []
        for array in self.testSet:
            A_testSet.append(self.add_hyn_sent(array))

        self.augmentSents = A_testSet

    def add_hyn_sent(self, array):
        A_array = []
        for word in array:
            x = wn.synsets(word)
            # indicates presence of hypernym
            if len(x) > 0:
                x = wn.synsets(word)[0]
                if x.hypernyms():
                    # synsets ordered by most prevalent
                    x = x.hypernyms()[0]
                    x = x.lemma_names()
                    A_array.append(x[0])
                A_array.append(word)
        return A_array

    def add_syn_sent(self, array):
        A_array = []
        for a in array:
            x = wn.synsets(a)
            if len(x) > 0:
                x = wn.synsets(a)[0]
                y = x.lemma_names()
                A_array.append(y[0])
            A_array.append(a)
        return A_array

    def augmentSent(self, array):
        # new array to hold sent
        A_array = []
        for a in array:
            # check if graph has term
            if self.graph.has_node(a):
                # if present find highest correlate
                term = self.augmentTerm(a)
                # append to new vector
                A_array.append(term)
            # append original term to vector
            A_array.append(a)

        # returned processed array
        return A_array

    def augmentTerm(self, term):
        # takes as argument term and finds graph compatriots
        lst = list(self.graph.edges(term, data = True))
        # orders list and returns top value
        lst.sort(key = lambda x: x[2]['weight'], reverse = True)

        # result is an ordered tuple set ((term, correlate, weight)...)
        # return top valued correlate
        return lst[0][1]

    def augmentTestSet(self, aug):

        # separates out a slice of testData
        self.returnTestData()

        if aug == "graph":
            print("process as graph")

            # slice equals the length of the data less the required sum for testing
            self.slice = len(self.dataList) - self.cut

            # populate graph with training data ; limit == cutoff point
            self.constructGraph(limit = self.slice)

            augmentSents = []
            print(len(self.testSet))
            for a in self.testSet:
                augmentSents.append(self.augmentSent(a))

            self.augmentSents = augmentSents


        if aug == "synonym":
            print("process as synonym")
            print(len(self.testSet))
            self.add_synonymns()

        if aug == "hypernym":
            print("process as hypernym")
            print(len(self.testSet))
            self.add_hypernyms()

        # make the sents NLTK friendly
        self.stringifyText()


    def build_tokenizer(self):
        "this is a function to a pruned sentence"
        token_patter = re.compile(token_pattern)

        return lambda doc: token_patter.findall(doc)

    def build_preprocessor(self):
        # preprocess text for tokenizer
        strip_accents = lambda x: x

        return lambda y: strip_accents(y.lower())

    def build_analyser(self):
        # call function to prune strings
        tokenise = self.build_tokenizer()

        # call processor to pass docs to processor
        preprocess = self.build_preprocessor()

        return lambda doc: self.remove_stopwords(tokenise(preprocess((self.decode(doc)))))

    def clusterMethod(self, cluster_list):

        # method takes in classlist and extracts docs and labels for clustering
        labels , docs = self.createListLabels(cluster_list)

        # initilise sklearn classifier
        vectoriser = TfidfVectorizer(max_df = 0.8, min_df = 0.2,
                                        use_idf = True, ngram_range = (1,3))
        # convert docs
        matrix = vectoriser.fit_transform(docs)

        km = KMeans(n_clusters = len(cluster_list))
        km.fit(matrix)
        clusters = km.labels_.tolist()



        clusters = pd.DataFrame({"clusters" : clusters, "actual_labels" : labels})

        pred = clusters.clusters.value_counts()

        pred.sort_index(inplace = True)
        cols = list(pred.index)

        df = pd.DataFrame(index = cols, columns = cols)

        for v in cols:
            f = clusters[(clusters.clusters == v)]
            act = f.actual_labels.value_counts()
            act.sort_index(inplace = True)
            df.iloc[v] = list(act)

        
        df['pred total'] = df.sum(axis= 1)
        df.loc['actual total'] = df.sum(axis= 0).T

        return df


    def createListLabels(self, classList):
        labels = []
        docs = []

        # loop over the classes and pull out information
        for i in range(len(classList)):
            docs.extend(classList[i].augmentSents)
            labs =  np.array([i]*len(classList[i].augmentSents))
            labels.extend(labs)

        return labels , docs

    def constructGraph(self , limit = None, depth = 2 ):
        # limit is a variable that allows one take a slice of the data to allow for testing/training split
        #
        g = nx.Graph()

        # iterate over docs saved to variables and create graph
        for array in self.dataList[:limit]:
            #print(len(array))
            dummyDepth = depth
            while len(array) > 1:
                g, array  = self.plotArray(array, dummyDepth, g)
                dummyDepth -= 1


        self.graph = g


    def decode(self, doc):
        """Decode the input into a string of unicode symbols

        The decoding strategy depends on the vectorizer parameters.
        """
        if isinstance(doc, bytes):
            doc = doc.decode('utf-8',  'strict')
            #doc = self.remove_stopwords(doc)

        return doc


    def plotArray(self, array, depth, g):
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


    def remove_stopwords(self, tokens):
        # returns tokenised list minus stop words
        tokens = [w for w in tokens if w not in stop]
        return tokens


    def returnPickle(self):
        try:
            df=  pd.read_pickle("data/" + self.name + ".pkl")
            self.df = df
        except:
            self.df = pd.DataFrame()

    def returnData(self):
        # method returns data
        try:
            df =  pd.read_csv("data/" + self.name + ".csv")
            return df.data
        except:
            "no file " + self.name + " to be found"
            return pd.DataFrame()

    def removeDuplicates(self, classList):
        # loop over input array and create one combined list
        all = []
        for csl in classList:
            all = all + list(csl.df[csl.df.columns[0]])

        # cast list to dataframe and identify duplicate values
        all = pd.DataFrame({"all": all})
        dubs = all.duplicated(keep = False)

        # pointers for monitoring which values correspond with which df
        shape = 0
        shape2 = 0
        # list to hold altered dataframes
        result = []
        # lopp over every frame
        for csl in classList:
            # mark start and finish point for reference in dubs
            shape2 = shape2 + csl.df.shape[0]
            x = list(dubs[shape : shape2])
            # invert the boolean values returned
            x = [not i for i in x]
            # appended df minus duplicate values
            result.append(csl.df[x])
            # update pointer to start of the df as seen in dubs
            shape = shape2

        return result


    def returnUpprocessedData(self, filename):
        location = "data/" + filename + ".pkl"
        #location = "data/rugby_text.csv"
        docs = []
        f = open(location)
        for line in f.read().strip().split(":-:"):
            docs.append(line)

        return docs

    def returnTestData(self):
        # method returns section of dataList not in the graph
        self.testSet = self.dataList[len(self.dataList) - self.cut:]

        # TODO put back in before testing
        # remove those items from the dataList
        self.dataList = self.dataList[:len(self.dataList) - self.cut]
        return self.testSet

    def selfie(self):
        return self

    def stringifyText(self):

        stringArray = []

        for array in self.augmentSents:
            stringy = ""
            for word in array:
                stringy = stringy + " " + word
            stringArray.append(stringy)

        self.augmentSents = stringArray




    def vectoriseData(self, process = False):
        try:
            df = self.df
            # initialise processor
            analyser = self.build_analyser()

            tester = []
            for data in df[df.columns[0]][:self.dataSlice]:
                d = analyser(data)
                # ensure uniformity minimum vector length
                if len(d) > vectorLength:
                    tester.append(d)

            self.dataList = tester

            return tester

        except:
            print("dataFrame is blank")
