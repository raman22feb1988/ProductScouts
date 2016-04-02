from __future__ import print_function
import os
import sys
from subprocess import check_call
from time import time
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.externals.six import StringIO
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.grid_search import GridSearchCV
import pydot 
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import matplotlib.pyplot as plt
from operator import itemgetter
from ggplot import *
import warnings
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

def get_data(filename,header_labels):
	data=pd.read_table(filename,header=None,names=header_labels)
	return data


product_headers=['product_title','brand_id','category']

#Data Preprocessing
product_data=get_data("classification_train.tsv",product_headers)
print(product_data)





sys.exit()


