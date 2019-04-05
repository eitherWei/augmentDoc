


from processingClass import *
import time

start = time.time()
testSize = 100
aug = "graph"
#aug = "synonym"
#aug = "hypernym"

# list of threads
data_list = ['red' , "rug", "ire", "pol", "wn"]
#data_list = ['red', 'rug', 'ire']

# initialise preprocessing class as pc
allD = []
for name in data_list:
    # initialise the class and process the input data
    pc = processingClass(name, testSize, dataSlice = 1000)
    allD.append(pc.selfie())

print(len(allD))

# remove any duplicates common to any of the five classes
classList = pc.removeDuplicates(allD)

# loop over the class lsit and assign the class df t that as processed above
for i in range(len(classList)):

    allD[i].df = classList[i]
    # vectorise and process the data
    allD[i].vectoriseData()

    # split the data into training and testing
    allD[i].returnTestData()

    # perform the required augmentation
    allD[i].augmentTestSet(aug)


clusters = pc.clusterMethod(allD)

print(clusters)





'''
# test set is taken from the datasection not on the Graph training set
pc.returnTestData()
# write the code that augments the test set
pc.augmentTestSet(aug)
'''
'''
f1 = 'AskReddit_text'
f2 = 'rugby_text'
f3 = 'ireland'
f4 = 'politics_text'
f5 = 'worldnews'
docs1 = pc.returnUpprocessedData(f1)[:100000]
docs2 = pc.returnUpprocessedData(f2)[:100000]
docs3 = pc.returnUpprocessedData(f3)[:100000]
docs4 = pc.returnUpprocessedData(f4)[:100000]
docs5 = pc.returnUpprocessedData(f5)[:100000]
#docs6 = pc.returnUpprocessedData(f6)
df1 = pd.DataFrame({"data" : docs1})
df2 = pd.DataFrame({"data" : docs2})
df3 = pd.DataFrame({"data" : docs3})
df4 = pd.DataFrame({"data" : docs4})
df5 = pd.DataFrame({"data" : docs5})
df1.to_pickle("red.pkl")
df2.to_pickle("rug.pkl")
df3.to_pickle("ire.pkl")
df4.to_pickle("pol.pkl")
df5.to_pickle("wn.pkl")
print(df1.shape)
print(len(docs1))
print(len(docs2))
print(len(docs3))
print(len(docs4))
print(len(docs5))
'''
'''
all = docs2 + docs4
df = pd.DataFrame({"all" : all})
x = df.duplicated() == True
#f2 = "/Volumes/Seagate Expansion Drive/testee.csv"
print(len(all))
df = pd.DataFrame({"all" : all})
x = df.duplicated()
test = df[x]
print(test.shape)
count = 0
for d in test['all']:
    d = d.split()
    if len(d) > 2:
        count = count + 1
print(count)
'''


print((time.time() - start)/60)
