import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import codecs
from nltk.corpus import stopwords
import warnings
from nlp_methods import nlpMethods
from  collections import Counter

warnings.filterwarnings('ignore')
stop = set(stopwords.words('english'))
stop.add("gt")
stop.add("et")
stop.add("al")


class mainMethods:

    def __init__(self, path):
        # tell the class where the directory is
        self.path = path
        # dictionary to hold all of our accroynms
        self.accronymList = {}
        # dataFrame that holds the information of the class
        self.df = pd.DataFrame()
        # dictionary to hold the phrases present
        self.phraseDict = {}
        # keep a record of the frequency of occurence of terms
        self.termsDict = {}
        # keep a records of the tfidf vectorizer
        self.tfidf_vectoriser = TfidfVectorizer()
        # # keep a records of the tfidf_matrix
        self.tfidf_matrix = pd.DataFrame()
        # store a count of all terms in the corpus
        self.termCorpusCount = {}
        # a list of all of the recorded dfs
        self.allTermIDFList = pd.DataFrame()
        # store a reference to words and dict locations
        self.wordStore = {}
        # pattern to change text area targeted
        self.pattern = "REFERENCES"

    def extractSections(self, text):
        ## method to extract sections
        sectionsArray = []
        # identifies sections
        for result in re.findall('<SECTION(.*?)</SECTION>', text, re.S):
            sectionsArray.append(result)

        # further extract headers (will be used as keys)
        sectionHeaders= []
        for section in sectionsArray:
            for result in re.findall('header=(.*?)>', section):
                sectionHeaders.append(result)

        # append key headers to text
        sectionDict = dict(zip(sectionHeaders, sectionsArray))
        return sectionDict


    def combineListofLists(self,  lista, listb):
        termsArray = []
        for terms in lista:
            if not isinstance(terms, int):
                termsArray.extend(terms)
        for terms in listb:
            if not isinstance(terms, int):
                termsArray.extend(terms)
        termsDict = dict(Counter(termsArray))
        return termsDict


    def returntermDictKeys(self, text):
        keys = list(text.keys())
        keysArray = []
        for key in keys:
            text = nlpMethods.lemmatise_corpus(self, re.sub('\"', "" , key.lower()))
            keysArray.append(text)

        return keysArray

    def returntermDictKeysNOT_lemmatised(self, text):
        keys = list(text.keys())
        keysArray = []
        for key in keys:
            text = re.sub('\"', "" , key.lower())
            keysArray.append(text)

        return keysArray

    def extractSectionContent(self, dataset, keyTerm):
        abstract = 0
        counter = 1
        contentList = []
        keyList = []
        for data in dataset.minimalCleanText:
            boolean = False
            for key, value in list(data.items()):
                prospect = key.lower().split()
                if len(prospect) == 1:
                    if keyTerm in prospect[0]:
                        abstract = abstract + 1
                        contentList.append(value)
                        keyList.append(key)
                        counter = counter + 1
                        boolean = True
            if not boolean:
                counter = counter + 1
                contentList.append("dud")
                keyList.append("dud")

        return keyList , contentList

    def extractRefernces(self, text):
        #pattern = 'REFERENCES'
        if text is not None:
            try:
                refs_loc = text.index(self.pattern)
                text1 = text[refs_loc:]
                return text1
            except:
                i = 1
            try:
                pattern1 = self.pattern.lower()
                pattern1 = pattern1[0].upper() + pattern1[1:]
                refs_loc = text.index(pattern1)
                text1 = text[refs_loc:]
                percent_loc  = float(refs_loc)/len(text)
                if percent_loc > .80:
                    return text1
            except:
                i = 1



    def lemmatiseCompTerms(self, text):
        sentArray = []
        # make sure that terms are present
        try:
            # split input text into array
            terms = text.split("\n")
            # loop over terms
            for term in terms:
                # phrases to be reduced to terms for lemmatising
                term = term.split()
                # ensure no blank strings
                if len(term) > 0:
                    # singletons
                    if len(term) == 1:
                        term = nlpMethods.lemmatise_corpus(self, term[0].lower())
                        sentArray.append(term)
                    # phrases
                    else:
                        termArray = []
                        # lemmatise each word in phrase
                        for t in term:
                            termArray.append(nlpMethods.lemmatise_corpus(self, t.lower()))
                        #remake phrase
                        term = " ".join(termArray)
                        sentArray.append(term)
        except:
            print("detected problem with below")
            print(text)
        return sentArray

    def extractIdfTermsDoc(self, text):
        #print(self.df.handle.index.values[text - 1])
        df1 = self.allTermIDFList[self.allTermIDFList.doc_id_list == (text- 1)]
        df1.sort_values(by=['term_idf_list'], inplace=True, ascending=False)
        tempDict = dict(zip(df1.term_list, df1.term_idf_list))

        return tempDict

    def amalgamateAllDocTermDictionaries(self, dict):
        self.termCorpusCount.update(dict)


    def extractAlternativeCorpus(self, path):
        #path = "Krapivin2009/all_docs_abstacts_refined/"
        _ , docs = self.extractFileNames(path)
        #print(docs)
        docLocations = []
        for item in docs:
            if ".txt" in item:
                path1 = path + item
                docLocations.append(path1)

        return docLocations

    def termCountDictionaryMethod(self, text):
        text = dict(Counter(text.split()))
        return text

    def countTotalTerms(self, text):
        # split the terms into dictionary counts
        text = text.split()
        return text

    def extractAccronymns(self):
        accroDict = {}
        for file in self.df.files:
            for f in file:
                acc , substring = self.extractCandidateSubstring(f)
                # make sure there are no empty strings or single letter matches
                if len(acc) > 1:
                    # store all accronyms
                    acc = acc.lower()
                    accroDict[acc] = substring

        # populate the class dictionary with accronyms from the corpus
        self.accronymList = accroDict

    def extractFileNames(self, path):
        listee = []
        dirs =  os.listdir(path)
        for d in dirs:
            path1 = path + d
            if(os.path.isdir(path1)):
                listee.append(int(d))
        listee.sort()
        # assigns the value to the class def
        self.df = pd.DataFrame({"handle" : listee})
        return listee , dirs

    def cleanData(self, data):

        cleaned = []
        for line in data:

            line = self.expandAcronyms(line)
            #remove urls
            line = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', line, flags=re.MULTILINE)
            # keep only text tokens
            line = re.sub('[^a-zA-Z]', ' ' , line)
            # break strings into an array
            line = line.split()
            # apply lower casing
            line = [x.lower() for x in line if x not in stop]
            # remove all single letters resulting from processing
            line = [x for x in line if len(x) > 1]
            # append to a list
            cleaned.extend(line)
        # convert list to string for processor friendly format
        cleaned = " ".join(cleaned)

        return cleaned

    # method for cleaning an array of string Terms
    def cleanTermsArray(self, array):
        clean_array = []
        # loop over array

        for a in array.split("\n"):
            # remove all non characters
            a = re.sub('[^a-zA-Z]', ' ' , a)
            clean_array.append(a.strip().lower())

        return clean_array

    def expandAcronyms(self, line):
        # identifier pattern for acronyms
        pattern = '\((.*?)\)'

        # cast the string to an array
        line = line.split(" ")
        # loop over the string looking for acronyms
        for i in range(0, len(line)):
            # extract terms inside brackets
            cand = re.search(pattern, line[i])
            # if there is a pattern match
            if cand:
                 # extract candidate
                candidate = cand.group(1)
                # check if candidate exists in established accronyms
                if candidate.lower() in self.accronymList:
                    # if yes update string with expanded accronym
                    accronym = " ".join(self.accronymList[candidate.lower()])
                    line.append(accronym)
        # return line as string
        line = " ".join(line)
        return line



    def extractCandidateSubstring(self, match):
        pattern = '\((.*?)\)'
        candidate = ""
        substring = ""
        #match = match.strip("\n")

        match = match.split(" ")
        for i in range(0, len(match)):
            cand = re.search(pattern, match[i])
            if cand:
                candidate = cand.group(1)
                # check that it is longer than 1
                if len(candidate) > 1:
                    # check and remove for non capital mix
                    if(self.lookingAtAcroynms(candidate)):
                        candidate = self.removeNonCapitals(candidate)
                    j = len(candidate)
                    substring = match[i-j:i]
                    # check if accronym is present
                    wordsAccro = self.returnPotentAccro(substring)
                    if candidate.lower() == wordsAccro.lower():
                        # return the correct accro and definition
                        return (candidate.lower(), substring)

        # no accronym found return blank , will be filtered out
        return("", "")

    def extractFiles(self, text):
        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        return self.path + str(text) + "/" + str(text) + ".txt"

    def extractXMLFiles(self, text):
        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        return self.path + str(text) + "/" + str(text) + ".xml"

    def extractKeyWordFiles(self, text):

        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        path = self.path + str(text) + "/" + str(text) + ".kwd"
        try:
            text = self.extractContent(path)
            #print(text)
        except:
            #print("keyword absent: " + str(text))
            x = 1
        return text

    def extractKeyWordFilesTerms(self, text):

        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        path = self.path + str(text) + "/" + str(text) + ".term"
        try:
            text = self.extractContent(path)
            text = self.cleanTermsArray(text)
            text = nlpMethods.lemmatise_corpus(self, " ".join(text))
            text = text.split()
        except:
            #print("keyword absent: " + str(text))
            x = 1
        return text

    def extractContent(self, text):
        #with open(text, "rb")   as file:
        with codecs.open(text, 'r', encoding='utf8', errors="ignore") as file:
            lines = file.read()
        return lines


    def applyTFidfToCorpus(self, dfList, title = "tfidf_store.pkl", failSafe = False):
        # create tf-idf matrix for the corpus
        #tfidf_matrix = None
        try:
            if (failSafe):
                ''' purposely crash try/except to force vectoriser rebuild '''
                x = 1/0

            print("-- Retrieving stored tfidf_matrix --")

            tfidf_matrix = pickle.load(open("matrix_" + title, "rb" ) )
            tfidf_vectoriser = pickle.load(open("vectorisor_" + title, "rb" ) )


        except:

            print("failed to load -- building tokeniser --")
            # initialise vectoriser and pass cleaned data
            tfidf_vectoriser = TfidfVectorizer(max_df = 0.9, min_df = 0.1, ngram_range = (1,4), stop_words ='english', tokenizer = self.tokenize_only)
            tfidf_matrix = tfidf_vectoriser.fit_transform(list(dfList))

            #df= pd.DataFrame({"tfidf_matrix" : tfidf_matrix}, index=[0])
            #save_tfidf.to_pickle("tfidf_min_04.pkl")
            #df.to_pickle("tfidf_matrix.pkl")

            # pickle tfidf matrix for faster future load
            with open("matrix_" + title, 'wb') as handle:
                        pickle.dump(tfidf_matrix, handle)

            # pickle tfidf vectoriser for faster future load
            with open("vectorisor_" + title, 'wb') as handle:
                        pickle.dump(tfidf_vectoriser, handle)

        return tfidf_matrix , tfidf_vectoriser


    def ExtractSalientTerms(self, tfidf_vectoriser, tfidf_matrix, title = "tfidf_.pkl",  failSafe = True):
        print('salient terms')
        df = pd.DataFrame()
        try:
            if (failSafe):
                ''' purposely crash try/except to force vectoriser rebuild '''
                x = 1/0

            print("loading presaved processed corpus --")
            df = pd.read_pickle(title)
            # lists for storing data

        except:
            print(" failed to load terms -- rebuilding -- ")
            doc_id_list = []
            term_list = []
            term_idf_list = []

            # extract terms from vectoriser
            terms = tfidf_vectoriser.vocabulary_


            keys = terms.keys()
            values = terms.values()

            # invert the dict so the keys are the values and values the keys
            dict1 = dict(zip(values, keys))

            # shortcut for saving and loading dictionary
            self.wordStore = dict1
            with open('dict' + title + '.pkl', 'wb') as f:
                pickle.dump(dict1, f, pickle.HIGHEST_PROTOCOL)




            # iterate through matrix
            for i in range(0, (tfidf_matrix.shape[0])):
                for j in range(0, len(tfidf_matrix[i].indices)):
                    # append the appropriate list with the appropriate value
                    doc_id_list.append(i)
                    term_list.append(dict1[tfidf_matrix[i].indices[j]])
                    term_idf_list.append(tfidf_matrix[i].data[j])

            # cast to dataframe
            df = pd.DataFrame({"doc_id_list": doc_id_list, "term_list" : term_list, "term_idf_list": term_idf_list})
            # pickle process for future fast retrieval
            df.to_pickle(title)

        print('loading dictionary')
        with open('dict' + title + '.pkl', 'rb') as f:
            self.wordStore =  pickle.load(f)

        print(list(self.wordStore.items())[:5])
        return df

    def extractTopNTerms(self, df ,  N = 10, title = "alt_termsList.pkl" , failSafe = False):
        # extract the terms specific to that document
        # list for storing document top terms
        try:
            if(failSafe):
                x = 1/0
            print("-- loading saved terms --")
            self.df['method_termDict'] = pd.read_pickle(title)

        except:
            print("failSafe -- building term list")
            termList = []
            for i in self.df.index:
                df1 = df[df.doc_id_list == i]
                # order the terms from highest to lowest
                df1.sort_values(by=['term_idf_list'], inplace=True, ascending=False)
                # extract the top N values
                values = list(df1.term_idf_list)[:N]
                # extract the top N terms
                terms =list(df1.term_list)[:N]
                # cast to dictionary
                termDict = dict(zip(terms, values))
                termList.append(termDict)

            # update the class df so that each doc has a corresponding termDict
            self.df['method_termDict'] = termList
            self.df['method_termDict'].to_pickle(title)

        return self.df['method_termDict']


    def tokenize_only(self, text):
        # data has been processed and requires only splitting into tokens
        tokens = [word for word in text.split(" ")]
        return tokens

    ################################################################################
    # code for looking at accroynms

    def extractCandidateSubstring(self, match):
        pattern = '\((.*?)\)'
        candidate = ""
        substring = ""
        #match = match.strip("\n")

        match = match.split(" ")
        for i in range(0, len(match)):
            cand = re.search(pattern, match[i])
            if cand:
                candidate = cand.group(1)
                # check that it is longer than 1
                if len(candidate) > 1:
                    # check and remove for non capital mix
                    if(self.lookingAtAcroynms(candidate)):
                        candidate = self.removeNonCapitals(candidate)
                    j = len(candidate)
                    substring = match[i-j:i]
                    # check if accronym is present
                    wordsAccro = self.returnPotentAccro(substring)
                    if candidate.lower() == wordsAccro.lower():
                        # return the correct accro and definition
                        return (candidate, substring)
        # no accronym found return blank , will be filtered out
        return("", "")

    # check of the main lettes match
    def returnPotentAccro(self, substring):
        firsts = ""
        for s in substring:
            if(len(s) > 0):
                firsts = firsts + s[0]
        return firsts


    def lookingAtAcroynms(self, accro):
        # case one check if accroynm has an append s
        bool = False
        for s in accro[:1]:
            if s.isupper:
                bool = True
        return bool

    def removeNonCapitals(self, accro):
        string = ""
        for s in accro:
            if s.isupper():
                string = string + s
        return string

    def extractCorpusPhraseArray(self, corpus, failSafe = False):
        try:
            if(failSafe):
                ''' purposely crash try/except to force phrase rebuild '''
                x = 1/0
            print("-- Retrieving stored phrase df --")
            df = pd.read_pickle("phraseDF.pkl")

        except:
            print("building -- extracting phrases from text -- ")
            # array to store each document dictionary
            dictionaryList = []
            for text in corpus:
                vector = self.createTextVectors(text)
                dict = self.extractDocPhraseArray(vector)
                dictionaryList.append(dict)

            df = pd.DataFrame({"phraseLists": dictionaryList})
            df.to_pickle("phraseDF.pkl")

        return df



    def extractDocPhraseArray(self, vector):
        doc_dict = {}
        index = 0
        sliding_window = 0
        term = ""
        #loop over the text vector
        while index < (len(vector)):
            if(index == sliding_window):
                term = vector[index]
            # if not in dict, add it
            if term not in doc_dict:
                # assign a value for the instance
                doc_dict[term] = 1
                # move the index forward
                index  = index + 1
                # reset the sliding_window
                sliding_window = index
            else:
                doc_dict[term] = doc_dict[term] + 1
                # increment the sliding window
                sliding_window = sliding_window + 1
                # create phrase term
                term = term + "_" +  vector[sliding_window]
            if sliding_window == len(vector) - 1:
                # take the sliding window back one to prevent outofbounderror
                sliding_window  = sliding_window - 1
                # increment index to move the term capture
                index = index + 1

        return doc_dict

    def createTextVectors(self, text):
        text_array = []
        # loop over each array in the text vector
        for t in text:
            # remove non characters
            t = re.sub('[^a-zA-Z]', ' ' , t)
            # split the strings into word vectors
            for t in t.split():
                # ignore any words less than 1
                if len(t) > 1:
                    # append the result to overall vector
                    text_array.append(t.lower())

        return text_array

    def removeIndex(df):
        for lines in df.files[0]:
            print(lines)
            print(10*"*")
