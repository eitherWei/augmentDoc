from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from nltk.corpus import wordnet as wn

from nltk.stem import WordNetLemmatizer



class nlpMethods:

    def __init__(self):
        self = self

    def evaluateTermResults(self, comp_terms, method_terms):
        #print(comp_terms)
        #print(method_terms)
        # array for storing each result
        allResults = []
        # loop over input and assess each prediction individually
        for i in range(len(method_terms)):
            # keywords missing from file ignore if not present
            if not isinstance(comp_terms[i], int):
                result = self.evaluateTermIndividually(comp_terms[i], method_terms[i])
                # store each result
                allResults.append(list(result))

        return allResults



    def evaluateTermIndividually(self, comp_terms, method_terms):
        #print(comp_terms)
        #print(method_terms)
        # create an array the size of the guesses equalling 1
        y_true = list(np.array([1]*len(method_terms)))

        # better with a lambda expression ?
        y_pred = self.ConvertPredictionsToNumerical(comp_terms, method_terms)

        # sklearn metric that returns (precision , recall, fscore _support)
        result = precision_recall_fscore_support(y_true, y_pred, average='binary')

        return result


    def ConvertPredictionsToNumerical(self, comp_terms, method_terms):
        results = []
        # loop over predict
        for t in method_terms:
            # loop over actual
            # determine success and mark accordingly
            if t in comp_terms:
                results.append(1)
            else:
                results.append(0)
        # return result equal to the input guesses
        return results

    def loadResultsDf(self, title):
        try:
            df = pd.read_pickle(title)
        except:
            df = pd.DataFrame()
        return df

    def plotResults(self, df):
        #print(df.head())
        title , values = self.extractResultValues(df, 2)
        i = 1
        x = list(range(len(values[0])))
        for i in range(len(values)):
            plt.plot(x, values[i])

        plt.xlabel("docs")
        plt.ylabel("results")
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.legend(['precision', 'recall', 'fscore'])
        plt.show()

    def extractResultValues(self, df, col):
        title = df.columns[col]
        #print(df.shape)
        #print(title)
        precision_array = []
        recall_array = []
        fscore_array = []
        # extract row values
        for row in df[df.columns[col]]:
            precision_array.append(row[0])
            recall_array.append(row[1])
            fscore_array.append(row[2])

        r = [precision_array, recall_array, fscore_array]
        return title, r

    def add_Hypernym(self, text):
        augmentArray = []
        for word in text.split():
            hyp = wn.synsets(word)
            # check if hypernym found
            if (hyp):
                hyp = wn.synsets(word)[0]
                if hyp.hypernyms():
                    # synsets ordered by relevance so take the first
                    hyp = hyp.hypernyms()[0]
                    hyp = hyp.lemma_names()
                    augmentArray.append(hyp[0])
            augmentArray.append(word)
        return " ".join(augmentArray)

    def lemmatise_corpus(self, text):
        lemmatiser = WordNetLemmatizer()
        termArray  = []
        for word in text.split():
            # lemmatise the term
            word = lemmatiser.lemmatize(word, pos='v')
            # remove plural instance
            word = lemmatiser.lemmatize(word)
            termArray.append(word)

        return " ".join(termArray)



            #print(hyp)
        #print(10*"*")
