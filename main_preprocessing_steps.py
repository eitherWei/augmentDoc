from preprocessMethods import *
from methods import *
import pandas as pd
import nltk
import time
from collections import Counter
start = time.time()

# read in target df
thread_id_one = "bb6w9y"
df = pd.read_pickle("data//" + thread_id_one + "_level_root.pkl")



# sanitiseData - removes puncutation , foreign terms , lowers, and stopword removal
tester = df.comment.apply(cleanData)

all_array = []
for array in tester:
    # creates one giant array
    all_array.extend(array)

# apply counter to the giant thread

wordset = Counter(all_array)
wordset.most_common()

temp_df = pd.DataFrame(wordset.keys())
temp_df.to_csv("checkWords.csv")

print(10*"-*-")
print(time.time() - start)
