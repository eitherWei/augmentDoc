# initialise the methods class

from  methods_main3 import *
from methods_main2 import *
import time
start = time.time()


crashIt = False
# open the dataClass
#path ="C:\\userOne\\keyPhraseExtraction-master\\keyPhraseExtraction-master\\AutomaticKeyphraseExtraction-master\\data\\"

methods_Krapivin = mainMethods(path)


# extract the file references
methods.extractFileNames(path)
methods_Krapivin.extractFileNames(otherPath)
# temp limit the graph to 2 files
#df_limit = methods.df[:10]
#methods.df = df_limit
print(methods.df.head())
print(methods_Krapivin.df.head())

def method_():
    # extract all file names associated with handles
    methods.df['fileNames'] = methods.df.handle.apply(methods.extractFiles)
    # extract the content
    methods.df['files'] = methods.df.fileNames.apply(methods.extractContent)
    # extract text
    methods.df['sanitiseData'] = methods.df.files.apply(methods1.cleanData)
    print("creating tfidf of terms")
    tfidf_matrix, tfidf_vectoriser = methods.applyTFidfToCorpus(methods.df.sanitiseData, failSafe = crashIt)
    # store as class variables
    methods.tfidf_matrix = tfidf_matrix
    methods.tfidf_vectoriser = tfidf_vectoriser
    # creates a dictionary of every term in the corpus
    methods.df['termsDict'] = methods.df.sanitiseData.apply(methods.termCountDictionaryMethod)
    # creates and overall dictionary count
    methods.df.termsDict.apply(methods.amalgamateAllDocTermDictionaries)
    # extract the keywords
    methods.df['keywords'] = methods.df.handle.apply(methods.extractKeyWordFiles)
    methods.df['competition_terms'] = methods.df.handle.apply(methods.extractKeyWordFilesTerms)

    # extract the tfidf and assign per document
    df = methods.ExtractSalientTerms(methods.tfidf_vectoriser, methods.tfidf_matrix, title = "tfidf_.pkl",  failSafe = crashIt)
    #df['idfTerms'] = methods.df.index.apply(extractIdfTermsDoc)
    methods.allTermIDFList = df
    methods.df['tfidf_list'] = methods.df.handle.apply(methods.extractIdfTermsDoc)



    methods.df.keywords = methods.df.keywords.apply(methods.lemmatiseCompTerms)
    ##############################################################################################
    # create document clusters

    ##############################################################################################
    ##############################################################################################
    # build graph
    #graph = methods1.plotCorpusToDiGraph(methods.df.sanitiseData)
    ##############################################################################################
    df = methods1.clusteringMethods(methods)
    df.to_pickle("ClusterAssignedDFS.pkl")




'''
df1 = pd.read_pickle("ClusterAssignedDFS.pkl")
print(df1.clusterAssignment)

from nltk.corpus import wordnet as wn
hyp = wn.synsets("test")
print(hyp)



i = 0
##############################################################################################
while i < 2:
    df1 = df1[df1.clusterAssignment == 1]

    d  , df1 = methods1.clusterMethodsRecursively(df1)

    d = dict(sorted(d.items(), key=lambda x: x[1]))

    print("the length of d is {}".format(len(d.items())))

    count = 0
    miss = 0
    for value in list(df1.keywords) or value in list(df1.competition_terms):
        for v in value:
            if v in d:
                print(v, d[v])
                count = count + 1
            else:
                miss = miss + 1

    print(count)
    print(miss)
    ##############################################################################################
    print(df1.shape)

    i = i + 1
'''
print(10*"===")

print(10*"-*-")
print((time.time() - start)/60)
