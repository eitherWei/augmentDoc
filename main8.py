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
