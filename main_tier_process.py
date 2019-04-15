import time
from preprocessMethods import *
import pandas as pd
start = time.time()

def update_dict(dict_ref, df_dummy):
    df_temp = df.id[df_dummy.index]
    for k , v in dict(df.parent[df_temp.index]).items():
        #print(dict_ref.keys())
        if v in dict_ref.keys():
            dict_ref[df_temp.loc[k]] = dict_ref[v]

    return dict_ref


def method_update_Dict( df, level):
    index_list = list(df.index)
    level_list = list(np.array([level]*len(list(index_list))))
    temp_dict = dict(zip(index_list, level_list))
    return temp_dict

def method_record_parents(level, df_temp, dict_ref):
    temp = {}
    if level == 0:
        index_list = list(df_temp.index)
        dict_ref = dict(zip(list(df.id[df_temp.index]), list(df.id[df_temp.index])))

    else:
        dict_ref = update_dict(dict_ref, df_temp)

    return(dict_ref)


#open the stored state of the thread
thread_id_one = "bbhpb5"
thread_id_one = "bb6w9y"

df = pd.read_pickle("data//" + thread_id_one + ".pkl")

#clean the thread_ids
df['id'] = df['id'].apply(swap)
df['parent'] = df['parent'].apply(swap)

thread_id = createContainsString([thread_id_one])

df_temp = df.copy()
dict_ref = {}
level = 0
dict1 = {}
level_dict = {}
while df_temp.shape[0] > 0:

    # create process the level dictionary
    t_1 = df['parent'].str.contains(thread_id)
    df_temp = t_1[t_1 == True]

    # process the parent strings
    dict_ref = method_record_parents(level, df_temp, dict_ref)

    if (len(thread_id) > 0) :
        temp_level_dict = method_update_Dict( df_temp, level)
        level_dict.update(temp_level_dict)
        dict_ref = method_record_parents(level, df_temp, dict_ref)

    # extract next wave of threads
    nextLevel = list(df[t_1 == True].id)
    thread_id = createContainsString(nextLevel)

    level  = level + 1


df['level'] = list(level_dict.values())
df['root_comment'] = list(dict_ref.values())


df.to_pickle("data//" + thread_id_one + "_level_root.pkl")


print(10*"-*-")
print(time.time() - start)
