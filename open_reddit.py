filename = "RC_2017-03"

import json
import pandas as pd

'''
df = pd.read_json(filename, lines = True)

print (df.columns)

print(df.shape)

df1 = list(df['subreddit'])

print(set(df1))
'''

list = []
with open(filename, 'r') as f:
    for i , line in enumerate(f):
        d = json.loads(line)
        if d['subreddit'].lower() == 'dublin':
            list.append(d)

df = pd.DataFrame(list)

df.to_pickle("dublin_mar.pkl")

import pandas as pd
df = pd.read_pickle("dublin_mar.pkl")
print(df.shape)
