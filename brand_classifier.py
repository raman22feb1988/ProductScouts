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
np.set_printoptions(threshold='nan')

def get_data(filename,header_labels):
	data=pd.read_table(filename,header=None,names=header_labels)
	return data


def align_datattypes(product_data):
	data_copy=product_data.copy()
	for row in data_copy:
		temp_category=row['category']
		row['category']=row['category'].tostr
	return data_copy	

product_headers=['product_title','brand_id','category']

#Data Preprocessing
product_data=get_data("classification_train.tsv",product_headers)
product_data['category']=np.array(product_data['category'],dtype=np.str_)
product_data['brand_id']=np.array(product_data['brand_id'],dtype=np.str_)

categories=product_data['category'].unique()
print(categories.size)
brands=product_data['brand_id'].unique()
print(brands.size)





sys.exit()


