from nlp_methods import *
from methods_main2 import *
import nltk
from nltk.stem import WordNetLemmatizer
lemmatiser = WordNetLemmatizer()
import time
start = time.time()
#nltk.download('averaged_perceptron_tagger')

path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
nlp_m = nlpMethods()
methods = mainMethods(path)


def calculateOverAllScore():
    # load up the results and get an overall score
    r = nlp_m.loadResultsDf("individual_results_df.pkl")
    result = [x[0] for x in r.tfidf_lemmatised]
    print(sum(result))

'''
calculateOverAllScore()
x = methods.extractKeyWordFilesTerms(1)
x =nlp_m.lemmatise_corpus( " ".join(x))
print(x)
'''
'''



# process terms
# method skips over blanks and loses index location
allTerms = []
for i in range(211):
    terms = methods.extractKeyWordFiles(i)
    if  not isinstance(terms, int):
        terms = methods.cleanTermsArray(terms)
        termsArray = []
        for term in terms:
            words = term.split()
            if len(words) > 1:
                phraseArray = []
                for word in words:
                    word = lemmatiseTerm(word)
                    phraseArray.append(word)
                phrase = " ".join(phraseArray)
            else:
                phrase = lemmatiseTerm(words[0])
            termsArray.append(phrase)
        allTerms.append(termsArray)

all_connections = []
for array in allTerms:
    tagArray = []
    for term in array:
        term = term.split()
        localTag = []
        if len(term)> 1 :
            pos = nltk.pos_tag(term)
            for i in range(len(pos)):
                localTag.append(pos[i][1])
        tagArray.append(localTag)
    all_connections.append(tagArray)


dictionary = {}
for connect in all_connections:
    for conn in connect:
        if len(conn) > 1:
            string = ""
            for item in conn:
                string = string + "_" + item
            if(string in dictionary):
                dictionary[string] = dictionary[string] + 1
            else:
                dictionary[string] = 1

dictionary = dict(sorted(dictionary.items(), key = lambda dictionary:(dictionary[1], dictionary[0]), reverse = False))


print(dictionary)

df_pattern = pd.DataFrame({"keys" : list(dictionary.keys()), "values" : list(dictionary.values())})
print(df_pattern.head())

df_pattern.to_pickle("df_pattern.pkl")

def PosTerms(term):
    count = 0
    term = nltk.pos_tag(term)
    for t in term:
        if "NN"  in t[1]:
            count = count + 1
    if count > 2:
        return True
    else:
        return False


    return (term)
def unpackTerms(text):
    text =  text.split("_")

    return text
def measureVectors(text):
    return (len(text))

#def removeNonNounPhrases(text):



df_pat = pd.read_pickle("df_pattern.pkl")
df = pd.read_pickle("phraseDF.pkl")

#keys = list(df.phraseLists[0].keys())
#values = list(df.phraseLists[0].values())
#df = pd.DataFrame({"keyss" : keys, "values" : values})

dictionary = {}
for d in df.phraseLists:
    dictionary.update(d)
df = pd.DataFrame({"keyss" : list(dictionary.keys()), "values" : list(dictionary.values())})


#df.sort_values(by=['values'], inplace=True, ascending=False)
df['phraseVectors'] = df.keyss.apply(unpackTerms)
df['lengths'] = df.phraseVectors.apply(measureVectors)
df = df[df.lengths > 1]

df['nounCandidates'] = df.phraseVectors.apply(PosTerms)

df = df[df.nounCandidates == True]
df.sort_values(by=['values'], inplace=True, ascending=False)
print(df)
print(df.shape)
print(10*"-*-")
print(time.time() - start)
