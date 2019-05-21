# read in the text
from methods_main2 import *
import codecs
import time

fileNum = 2
start = time.time()
path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/2/2.txt"
#path = /Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/2/2.txt
methods = mainMethods(path)
x = methods.extractKeyWordFilesTerms(2)

with codecs.open(path, 'r', encoding='utf8', errors="ignore") as file:
        lines = file.read()

def Find(string):
    # findall() has been used
    # with valid conditions for urls in string
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
    return url

line = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', line, flags=re.MULTILINE)


print(len(lines))
print(lines.find("REFERENCES"))

path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"


'''
array = lines.split(". ")
for line in array:
    print(line)
    print(10*"-x-")
print(x)

terms = methods.extractKeyWordFiles(2)
print(terms)

df = pd.read_pickle("tfidf_whole_corpus.pkl")
df = df[df.doc_id_list == 2]
print(df.head())
df.sort_values(by=['term_idf_list'], inplace=True, ascending=False)
print(df.head())
# looking at the features of the sentence
#def sentFeats(sentence, index):


'''

print(time.time() - start)
