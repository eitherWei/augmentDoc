filename = "RC_2017-03"

#https://github.com/alumbreras/reddit_parser <-- good tutorial on  parsing in a stream


import json
import ijson
import pandas as pd
import time

start = time.time()
list = []

with open(filename, 'r') as f:
    for i , line in enumerate(f):
        d = json.loads(line)
        if d['subreddit'] == 'worldnews':
            list.append(d)


df = pd.DataFrame(list)

print(df.keys())

df.to_pickle("world_news_mar.pkl")

df1 = pd.read_pickle("world_news_mar.pkl")
print(df1.head())
print((time.time() - start)/60)
