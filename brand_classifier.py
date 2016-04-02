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
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
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

# sample_text="120GB Hard Disk Drive with 3 Years Warranty for Lenovo Essential B570 Laptop Notebook HDD Computer - Certified 3 Years Warranty from Seifelden"
sample_text="Sony VAIO VPC-CA4S1E/W 14.0"" LCD LED Screen Display Panel WXGA HD Slim	415 CANON USA IMAGECLASS D550 - MULTIFUNCTION - MONOCHROME - LASER - PRINT, COPY, SCAN - UP TO 4509B061AA	274 Monoprice 104234 MPI Dell Color Laser 3010CN - Black with Chip	3 Dell Ultrabook XPS 12 Compatible Laptop Power DC Adapter Car Charger	658 ProCurve Switch 4208vl U.S ProCurve Networking 6H - J8773A#ABA	437 Dell PowerEdge R710 - 1 x X5650 - 16GB - 5 x 600GB 10K	248"
text=word_tokenize(sample_text)
print(nltk.pos_tag(text))
tagged_text=pos_tag(sample_text.split())
print(tagged_text)
print(nltk.ne_chunk(tagged_text, binary=True))
print(nltk.ne_chunk(tagged_text))

sys.exit()


