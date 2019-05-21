# reading in docs from xml
#import xml.etree.ElementTree as et
import pandas as pd
from methods_main2 import *
from methods_main3 import *
from bs4 import BeautifulSoup
import xml
import re
import time

start = time.time()
# df for dataset
dataset = pd.DataFrame()
# initialise methods class
path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
methods = mainMethods(path)

# extracts the files handles
methods.extractFileNames(path)
# load up full filepath (xml)
methods.df['fileNames'] = methods.df.handle.apply(methods.extractXMLFiles)
methods.df['files'] = methods.df.fileNames.apply(methods.extractContent)

### assigning sectionDictionary to dataset
dataset['sectionsDict'] = methods.df.files.apply(methods.extractSections)

# lowercase and punctuation \n removal
minimalCLeanlist = []
for i in range(211):
    minimalCleanText = {}
    for k, v in dataset.sectionsDict[i].items():
        newString = "".join(v.split("."))
        newString = "".join(newString.split(","))
        newString = "".join(newString.split("'"))
        newString = "".join(newString.split("\""))
        newString = "".join(newString.split(":"))
        newString = "".join(newString.split(";"))
        #s = re.sub('\n',' ', s)
        minimalCleanText[k] = newString.lower()
    minimalCLeanlist.append(minimalCleanText)
dataset['minimalCleanText'] = minimalCLeanlist

#print(dataset.minimalCleanText[0]['"ABSTRACT"'])
#print(10*"-=_")
#print(dataset.sectionsDict[0]['"ABSTRACT"'])


keyList_ab, dataset['abstract'] = methods.extractSectionContent(dataset, 'abstract')
keyList_intro, dataset['introduction'] = methods.extractSectionContent(dataset, 'introduction')
keyList_ref, dataset['references'] = methods.extractSectionContent(dataset, 'reference')


########################################################
# extracting all dict content not in above sections
dud = keyList_intro.count("dud")
dud = keyList_ab.count("dud")
dud = keyList_ref.count("dud")

trueList = []
for i in range(211):
    boolean = True
    if keyList_intro[i] == "dud":
        boolean = False
    if keyList_ab[i] == "dud":
        boolean = False
    if keyList_ref[i] == "dud":
        boolean = False
    trueList.append(boolean)

dataset['trueList'] = trueList

dict = dataset.minimalCleanText

for i in range(211):
    if dataset.trueList[i] == True:
        del dict[i][keyList_intro[i]]
        del dict[i][keyList_ab[i]]
        del dict[i][keyList_ref[i]]

dataset['otherDict'] = dict

# loop over keys and values and make one giant 'other' DOCUMENT
otherList = []
for i in range(1):
    othersDoc = ""
    for key , value in dataset.otherDict.items():
        for k , v in value.items():
            othersDoc = othersDoc + " " + v
        otherList.append(othersDoc)

print(len(otherList))

dataset['otherSections'] = otherList

print(dataset.columns)


########################################################
# clean all of the tile headers
dataset['cleanHeaders'] = dataset.sectionsDict.apply(methods.returntermDictKeys)


print(10*"-->")
# iterate over headers and calculate hit factor
# loads the target terms per document
terms_df = pd.read_pickle("compTermInstance.pkl")
dataset['targetTerms'] = terms_df.presentTerms


# iterate over the two rows and find the overlapping lads
# this is done on a lemmatised corpus
headerCount = 0
for i in range(dataset.targetTerms.shape[0]):
    overlap = list(set(dataset.targetTerms[i]).intersection(dataset.cleanHeaders[i]))
    #print(overlap)
    headerCount = headerCount + len(overlap)


print(" There are {} keyphrases present in Headers".format(headerCount))


# another way of achieving above
headerCount = 0
for i in range(dataset.targetTerms.shape[0]):
    for term in dataset.targetTerms[i]:
        if term in dataset.cleanHeaders[i]:
            headerCount = headerCount + 1


def convertRawTermsToDict(text):
    if not isinstance(text, int):
        text = text.split("\n")
        text = [x.lower() for x in text if len(x) > 0]
    return text


#  lets make a comparison without processing the text
# extract the raw terms and add them to our reference df
terms_df['raw_keywords'] = terms_df.handle.apply(methods.extractKeyWordFiles)
terms_df['list_raw_competition_terms'] = terms_df.handle.apply(methods.extractKeyWordFilesTerms)
# lower casing and removing shite
terms_df['list_raw_keywords'] = terms_df.raw_keywords.apply(convertRawTermsToDict)
termsDict = methods.combineListofLists(list(terms_df['list_raw_competition_terms']), list(terms_df['list_raw_keywords']))

combinedList = []
for i in range(dataset.targetTerms.shape[0]):
    text1 = terms_df.list_raw_competition_terms[i]
    text2 = terms_df.list_raw_keywords[i]
    if isinstance(text1,  int):
        text1 = [str(text1) + "_duddee"]
    if isinstance(text2,  int):
        text2 = [str(text2) + "_duddee"]
    listee = text1 + text2
    combinedList.append(listee)


dataset['cleanHeadersNOTLemmatised'] = dataset.sectionsDict.apply(methods.returntermDictKeysNOT_lemmatised)

dataset['keywordsNotLemmatised']  = combinedList


headerCount1 = 0
overlapList = []
for i in range(dataset.targetTerms.shape[0]):
    overlap = list(set(dataset.cleanHeadersNOTLemmatised[i]).intersection(dataset.keywordsNotLemmatised[i] ))
    print(overlap)
    headerCount1 = headerCount1 + len(overlap)
    overlapList.append(overlap)


#  There are 28 keyphrases present in Headers

########################################################
# extract number of phrases in the extract
#print(dataset.abstract.head(10))


print(dataset.columns)
abstractCount = 0
for i in range(211):
    for term in dataset.keywordsNotLemmatised[i]:
        term1 = str(term)
        #print(dataset.references[i])
        if term1 in dataset.otherSections[i]:
            #print(term1)
            abstractCount = abstractCount + 1
print(" There are {} keyphrases present in introduction".format(abstractCount))
#  There are 28 keyphrases present in Headers - lemmatised == NO
#  There are 513 keyphrases present in abstract - lemmatised == NO
# There are 652 keyphrases present in introduction - lemmatised == NO
#  There are 567 keyphrases present in references - lemmatised == NO
#  There are 1095 keyphrases present in Others - lemmatised == NO
# total

'''
########################################################
## to determine what keyphrases exist in the rest of the text
#

print(dataset.shape)
print(len(keyList_ref))
print(len(keyList_ab))
print(len(keyList_intro))

print(dataset.columns)

i = 0

target = list(dataset.sectionsDict[i])
abs = keyList_ab[i]
ref = keyList_ref[i]
intro = keyList_intro[i]

print(target)
print(abs)

print(dataset.columns)


trueList = []
for i in range(211):
    target = list(dataset.cleanHeaders[i])
    abs = keyList_ab[i]
    boolean = True
    if abs.lower() not in target:
        boolean = False
        print(abs.lower())
        print(target)
    trueList.append(boolean)

print(trueList.count(True))
print(trueList.count(False))

dataset['trueList'] = trueList

tester = dataset[dataset.trueList == True]
falseList = dataset[dataset.trueList == False]

print(tester.shape)
print(falseList.shape)
'''
########################################################

######
### extracting sections
# ---------------------
# section[0] == abstract
# 101 omitted due to not having an abstract
# 111 does not contain an abstract
## 
#section[1 or 2 or 3] == introduction  - 205 instances
# - 0 : in one instance
# case 48 == background
# case 62 == THE PROBLEM
# case 99 == Technology overview
# case 107 == One Language, Many Languages
# case 109 == MOTIVATION
# case 113 == EVOLUTION OF ONLINE NETWORKING
##
# 170 conclusions in general search -- skip
#
##
# references found twice in 3 papers -> careful


# case 1 : "BIBLIOGRAPHY"

print(10*"-*-")
print((time.time() - start)/60)
#xtree = et.parse(methods.df.fileNames[0])
