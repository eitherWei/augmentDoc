import os

def extractFileNames(path):
    fileNames = []
    for r, d, f in os.walk(path):
        for file in f:
            fileNames.append(file)

    return fileNames

def seperateBySuffix(file_list , suffix):
    #loop over filelist and identify target docs
    target_list = []
    for filename in file_list:
        if  suffix in filename:
            target_list.append(filename)
    return target_list
