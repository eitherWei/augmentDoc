from methods_main2 import *
from methods_main3 import *
import time
start = time.time()

path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
methods = mainMethods(path)
methods1 = processMethods()
#path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/2/2.txt"
#lines = methods1.extractContent(path)
#print(len(lines))


## tfidf_matrix == location of initial data store
## vectorisor and corpus == "tfidf_whole_corpus.pkl"
crashIt = True


#############################################################
# processing alt content

otherPath = "Krapivin2009/all_docs_abstacts_refined"


def extractAlternativeCorpus():
    path = "Krapivin2009/all_docs_abstacts_refined/"
    _ , docs = methods.extractFileNames(path)

    docLocations = []
    for item in docs:
        if ".txt" in item:
            path1 = path + item
            docLocations.append(path1)

    return docLocations
'''
alt_corpus =  extractAlternativeCorpus()
df_alt = pd.DataFrame({"fileNames" : alt_corpus})
df_alt['files'] = df_alt.fileNames.apply(methods1.extractContent)
print("cleaning text....")
df_alt['sanitisedData'] = df_alt.files.apply(methods1.cleanData)
print(df_alt.head())

df_alt.to_pickle("altContentStored.pkl")
'''
df_alt = pd.read_pickle("altContentStored.pkl")
tfidf_matrix, tfidf_vectoriser = methods.applyTFidfToCorpus(df_alt.sanitisedData, title = "alt_tdfidf_store.pkl", failSafe = crashIt)
print(tfidf_matrix.shape)

# extract a df of all of the rated terms per document
df_terms = methods.ExtractSalientTerms(tfidf_vectoriser, tfidf_matrix, title = "alt_tfidf_whole_corpus.pkl", failSafe = crashIt)
# order the df and assgin the class df with the top N terms per document
#df  = methods.extractTopNTerms(df_terms , title = "alt_terms_list.pkl" N = 20, failSafe = crashIt)

df = pd.read_pickle("alt_tfidf_whole_corpus.pkl")

df1 = df[df.doc_id_list == 0]
print(df1.shape)

g = methods1.plotCorpusToDiGraph(df_alt.sanitisedData[0].split(), "corpusGraph3", failSafe = True)
#print(g.nodes())
print(g['an'])

def extractPhrases():
    print("extracting phrases")
    #print(len(methods.df.sanitiseData))
    doc = df_alt.sanitisedData[0]
    doc = doc.split()

    for i in range(len(doc) - 1):
        #print(doc[i])
        #print(i)
        #print(doc[i], doc[i+1])
        if g.has_edge(doc[i], doc[i + 1]):
            print(true)
            #print(doc[i], g.has_edge(doc[i], doc[i+ 1]), g[doc[i]][doc[i+ 1]]['weight'])
            prob_A_B = g[doc[i]][doc[i+ 1]]['weight']

            likelihood= methods1.calculate_prob_a_given_b(lookUp[doc[i]], lookUp[doc[i + 1]], prob_A_B)
            print(doc[i], doc[i + 1], likelihood)
            #except:
                #print(doc[i], doc[i + 1], "absent")
            #    x = 1


#extractPhrases()




'''
#############################################################



#############################################################
print("extracing corpus text...")
# extract the available files
files = methods.extractFileNames(path)
# assign the file reference numbers to the df
methods.df = pd.DataFrame({"handle" : files[:2]})
# extract all file names associated with handles
methods.df['fileNames'] = methods.df.handle.apply(methods.extractFiles)
# extract the content
methods.df['files'] = methods.df.fileNames.apply(methods.extractContent)
# extract text
methods.df['sanitiseData'] = methods.df.files.apply(methods1.cleanData)
#############################################################
termString = []
for doc in methods.df.sanitiseData:
    doc = doc.split()
    termString.extend(doc)
#############################################################

print(len(set(termString)))


print("processing text")
# create tf_idf matrix
# create/load a df_idf ranking of the corpus
print("creating tfidf of terms")
tfidf_matrix, tfidf_vectoriser = methods.applyTFidfToCorpus(methods.df.sanitiseData, failSafe = crashIt)
print(tfidf_matrix.shape)
'''
'''
# extract a df of all of the rated terms per document
df_terms = methods.ExtractSalientTerms(tfidf_vectoriser, tfidf_matrix, failSafe = crashIt)
# order the df and assgin the class df with the top N terms per document
df  = methods.extractTopNTerms(df_terms ,  N = 20, failSafe = crashIt)

#print(tfidf_matrix)

################################
# keyTerms
#methods.df['keywords'] = methods.df.handle.apply(methods.extractKeyWordFiles)
#methods.df['competition_terms'] = methods.df.handle.apply(methods.extractKeyWordFilesTerms)
#fileNum = 6
#print(methods.df['competition_terms'][fileNum])
#print(methods.df.keywords[fileNum])
#print(df[fileNum])
################################

#methods.df['Phrases'] = methods.df.files.apply(methods1.extractPhrases)
#for file in methods.df.files:
#file = methods.df.files[fileNum]
#print(file)
#print(len(g))
#print(g['boolean'])

#methods['cleanData'] = methods1.df.file.apply()
'''
'''

#g = methods1.plotCorpusToDiGraph(methods.df.semiProcesseData, "corpusGraph", failSafe = False)

g = methods1.plotCorpusToDiGraph(methods.df.sanitiseData, "corpusGraph2", failSafe = crashIt)

print(len(g))
print(g['boolean'])
#33659

df = pd.read_pickle("tfidf_whole_corpus.pkl")

df1 = df[df.doc_id_list == 0]
df2 = df[df.doc_id_list == 1]
lookUp = dict(zip(df1.term_list, df1.term_idf_list))
#lookUp1 = dict(zip(df1.term_list, df1.term_idf_list))
#print(df.head())
print(len(lookUp.keys()))

#print(df.term_list)
print()

graphTerms = g.nodes()
print(19*"-")
for terms in df.term_list:
    if terms not in list(g.nodes()):
        print(terms)

count = 0
doc = methods.df.sanitiseData[0]
for term in doc.split():
    if term not in list(df.term_list):
        count = count + 1

print(count)
def extractPhrases():
    print("extracting phrases")
    #print(len(methods.df.sanitiseData))
    doc = methods.df.sanitiseData[0]
    #print(doc)
    doc = doc.split()
    for i in range(len(doc) - 1):
        #print(doc[i])
        if g.has_edge(doc[i], doc[i + 1]):
            #print(doc[i], g.has_edge(doc[i], doc[i+ 1]), g[doc[i]][doc[i+ 1]]['weight'])
            prob_A_B = g[doc[i]][doc[i+ 1]]['weight']

            likelihood= methods1.calculate_prob_a_given_b(lookUp[doc[i]], lookUp[doc[i + 1]], prob_A_B)
            print(doc[i], doc[i + 1], likelihood)
            #except:
                #print(doc[i], doc[i + 1], "absent")
            #    x = 1


#fileNum = 0
#methods.df['competition_terms'] = methods.df.handle.apply(methods.extractKeyWordFilesTerms)
#methods.df['keywords'] = methods.df.handle.apply(methods.extractKeyWordFiles)
#print(methods.df.keywords[fileNum])
#print(methods.df['competition_terms'][fileNum])

extractPhrases()
'''
'''
# check if we can do the extract trick
xx = []
yy = []
for files in methods.df.files:
    x = files.find("REFERENCES")
    xx.append(x)
    y = files.find("references")
    yy.append(y)

df =  pd.DataFrame({"a" : xx, "b" : yy})
df = df[df.a != df.b]
print(df)
'''
print(10*"-**-")
print((time.time() - start)/60)
