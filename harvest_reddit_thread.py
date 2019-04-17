import praw
from requests import Session
from pprint import pprint
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from praw.models import MoreComments
import numpy as np
import re
import sys

start = time.time()



r = praw.Reddit(client_id='', \
                     client_secret='', \
                     user_agent='', \
                     username='', \
                     password='')

#subreddit = r.subreddit('learnpython')
#for submissions in subreddit.hot(limit = 10):
#    print(submissions)


thread_id = "bb6w9y"
'''
sub = r.submission(id=thread_id)
pprint(vars(sub))
print(sub.title)
print(sub.author)
print(sub.id)
print(sub.score)
print(sub.created)
upratio = sub.upvote_ratio
topcommsCnt = len(sub.comments)
allcommsCnt = len(sub.comments.list())
print(upratio)
print(topcommsCnt)
print(allcommsCnt)


print("1st entry is:")
print(r.submission(id=thread_id).title )
'''


def collectSubData(submission):
    post = r.submission(id=submission) #Access subreddit post based on submission id
    subData = list() #list to store key data of submission
    title = post.title
    url = post.url
    flair = post.flair
    author = post.author
    unique = post.id
    score = post.score
    created = datetime.datetime.fromtimestamp(post.created) #e.g. datetime
    upratio = post.upvote_ratio
    topcommsCnt = len(post.comments)
    allcommsCnt = len(post.comments.list())
    #parent = post.parent_id


    #subData.append((unique,title,url,author,score,created,upratio,topcommsCnt,allcommsCnt,flair))
    headers = ["Post ID","Title","Url","Author","Score","Publish Date","Upvote Ratio","Total No. of Top Comments","Total No. of Comments2","Flair"]
    data1 = [unique, title, url, author, score, created, upratio, topcommsCnt, allcommsCnt, flair]

    #dict = zip(headers, data)

    #df = pd.DataFrame.from_dict(data1, columns = headers)
    df = pd.DataFrame([data1], columns=headers)
    #print(df.columns)
    #print(df.head())

    return post , df




def retrieve_all_comments(post):
    #post.comments.replace_more(limit = 0)
    headers = ['parent', 'comment', 'id', 'author', 'created', 'score']
    DictArray = []
    post.comments.replace_more(limit=None)
    for comment in post.comments.list():
        oneList = []
        oneList.append(comment.parent_id)
        oneList.append(comment.body)
        oneList.append(comment.id)
        oneList.append(comment.author)
        oneList.append(comment.created_utc)
        oneList.append(comment.score)
        dict = zip(headers, oneList)
        DictArray.append(oneList)


    df = pd.DataFrame(DictArray, columns=headers)
    print(df.shape)

    return df

def tierExtract(df, id, level):
    #re.sub(r'.*_', '', id)
    #print(id)

    t_1 = df['parent'].str.contains(id)

    return  t_1

def createContainsString(array):
    query = ""
    for a in array:
        query = query + a+ "|"

    #print(query[:-1])
    return query[:-1]

## replace author object with string
def swap(text):
    up = re.sub(r'.*_', '', text)
    return up

def parentRef(dict1  , input_id, input_parent):
    if input_parent in dict1:
        dict1[input_id] = input_parent
    else:
        print("this should not appear")

    return dict1


thread_id = "bbhpb5"

t = False
if(t):
    post , data  = collectSubData(thread_id)
    df = retrieve_all_comments(post)
    df.to_pickle("data//" + thread_id + ".pkl")



#print(df.shape)
df = pd.read_pickle("data//" + thread_id + ".pkl")

df['id'] = df['id'].apply(swap)
df['parent'] = df['parent'].apply(swap)




#####################
#level 0
print("level 0")
thread_id = createContainsString([thread_id])
t_1 = tierExtract(df, thread_id, 0)

df_level = df[t_1 == False]
print(df_level)

df_dummy = df[t_1]
index_list = list(df_dummy.index)
#id_list = list(df_dummy.id)

level = 0
level_list =  list(np.array([level]*len(list(index_list))))
dict1 = dict(zip(index_list, level_list ))


dict_ref = dict(zip(list(df_dummy.id), list(df_dummy.id)))
print(len(dict_ref.items()))

print(df_level)


#####################


#####################

#level 1
print("level 1")
nextLevel = list(df[t_1 == True].id)
print(nextLevel)
print(20*"^")
thread_id = createContainsString(nextLevel)
print(thread_id)
t_1 = tierExtract(df_level, thread_id, 1)

print(t_1)
'''
level = 1
df_dummy = df_level[t_1]
index_list = list(df_dummy.index)
level_list =  list(np.array([level]*len(list(index_list))))
dumy_dict = dict(zip(index_list, level_list ))
dict1.update(dumy_dict)
#print(dict1)



def update_dict(dict_ref, df_dummy):
    for k , v in dict(df_dummy.parent).items():
        print(k , v)
        if v in dict_ref.keys():
            dict_ref[df_dummy.loc[k].id] = dict_ref[v]

    return dict_ref

dict_ref = update_dict(dict_ref, df_dummy)
print(len(dict_ref.items()))

print(df.comment)

#parent_id_df = df_dummy['id', 'parent']
#print(parent_id_df)
#parent(dict_ref  , input_id, input_parent):


#####################

#####################
#level *
print("level 2")

df_level_2 = df_level[t_1 == False]
nextLevel = list(df_level[t_1 == True].id)
thread_id = createContainsString(nextLevel)
print(thread_id)
t_1 = tierExtract(df_level_2, thread_id, 2)


df_dummy = df_level_2[t_1]
df_level_3 = df_level_2[t_1 == False]


dict_ref = update_dict(dict_ref, df_dummy)
print(len(dict_ref.items()))


df_level_3 = df_level_2[t_1 == False]
nextLevel = list(df_level_2[t_1 == True].id)
thread_id = createContainsString(nextLevel)




t_1 = tierExtract(df_level_3, thread_id, 3)


df_dummy = df_level_3[t_1]



dict_ref = update_dict(dict_ref, df_dummy)

print("length")
print(len(dict_ref.items()))


checker = list(dict_ref.values())

for k , v in dict_ref.items():
    print(k, v)


'''




'''
print(t_1)
# sort the values and shave off the most common
parent_count = t_1.groupby('parent').count()
parent_count = parent_count.sort_values(by='comment', ascending = False)
#print(parent_count.index[0])



# determine potential parent roots
prospects = list(parent_count.index)
cleaned_prospects = []
for p in prospects:
    item = re.sub(r'.*_', '', p)
    cleaned_prospects.append(item)

# remove prefix
print("prefixes removed")
print(cleaned_prospects)

print(t_1.id)
print(df.id)


#print(df)


print(df.columns)
#print(df.describe())
print(df.shape)
df_id = [[df.parent, df.id, df.comment]]
print(df_id)

t_1 = df['parent'].str.contains(thread_id)
t_1 = df[t_1==False]
# count the number of parents they contain
print(t_1)
parent_count = t_1[['parent','id']].groupby('parent').count()
#parent_count = parent_count.sort_values(by='parent', ascending = False)

print("---")
#print(t_1.parent)
print(type(parent_count))


## replace author object with string
def swap(text):
    data = "[deleted]"
    if text is not None:
        data = text.name
    #print(vars(text))
    #new = text.replace("", df.author.name)
    return data

def drillExtract(text):
    t1_list = []
    if "t1_" in text:
        t1_list.append(text)
        print(text)


df['author'] = df['author'].apply(swap)

#print(type(df.author[0].name))
authors_count = df[['comment','author']].groupby('author').count()


authors_count = authors_count.sort_values(by='comment', ascending=False)
print(authors_count)

publisher_scores = df[['score','author']].groupby('author').sum()
publisher_scores = publisher_scores.sort_values(by='score', ascending = False)


print(publisher_scores.head(15))

print(10*"*")
top_comment = df[df.parent == df.id]
#parent_count = parent_count.sort_values(by='parent', ascending=False)
print(top_comment)

print(df.parent.head())
print(df.id.head())

#lets get an eyeball on how many levels there are
print(df.shape)
# there are a total of 2195 comments
# iterate over this list and determine how many t*_ there are
t1List = df['parent'].str.contains('t1_')
df_t1  = df[t1List]
print(df_t1.shape)
# t1 == 1926
# t2 == 0
# t3 == 296
print(10*"*")
tList = df['parent'].str.contains('t3_')
df_t2  = df[tList]
print(df_t2.shape)
'''


#print(df['parent'])
print((time.time() - start)/60)


#game_threads = data[data["Title"].str.contains('Showcase|Megathread|Discussion|Day|Conference')==False]
#tester = post.comments.list()[0]
#print(vars(tester))
