from __future__ import print_function
import os
import sys
import re
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

def find_catalog(product_name,catalog_of_products):
	temp_catalog=list(catalog_of_products)
	tagged_text=pos_tag(product_name.split())
	output=nltk.ne_chunk(tagged_text)
	for subtree in output.subtrees(filter=lambda t: t.label() == 'PERSON'):
		for leave in subtree.leaves():
			temp_catalog.append(leave[0])
	return temp_catalog	


def run_rules(product_title):
	if(re.match("^decal|matte",product_title,re.IGNORECASE)):
		return 26992
	elif(re.match("^sandisk",product_title,re.IGNORECASE)):
		return 30503
	elif(re.match("^rikki",product_title,re.IGNORECASE)):
		return 36274
	elif(re.match("^ibm",product_title,re.IGNORECASE)):
		return 4361
	elif(re.match("^lb1",product_title,re.IGNORECASE)):
		return 22765
	elif(re.match("^first2savvv",product_title,re.IGNORECASE)):
		return 31194
	elif(re.match("^zectron",product_title,re.IGNORECASE)):
		return 11544								
	if((re.search("card",product_title)) or (re.search("adapter",product_title,re.IGNORECASE))):
		return 11580
	if((re.search("generic",product_title)) or (re.search("for",product_title)) or (re.search("Generic",product_title)) ):
		return 6584
	elif(re.search("HP",product_title)):
		return 42935
	return 45688		

	# HP Rule
	# try:
	# 	if (product_title.index("generic")>=0) or (product_title.index("for")>=0):
	# 		return 6584
	# 	elif(product_title.index("HP")>=0):
	# 		return 42835
			
	
	# except ValueError:
	# 	return 42836	

		

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
product_data=product_data.drop_duplicates()
print("Removed the Duplicates....")

# categories=product_data['category'].unique()
# print(categories.size)
# brands=product_data['brand_id'].unique()
# print(brands.size)

# sample_text="120GB Hard Disk Drive with 3 Years Warranty for Lenovo Essential B570 Laptop Notebook HDD Computer - Certified 3 Years Warranty from Seifelden"
# sample_text="Sony VAIO VPC-CA4S1E/W 14.0"" LCD LED Screen Display Panel WXGA HD Slim	415 CANON USA IMAGECLASS D550 - MULTIFUNCTION - MONOCHROME - LASER - PRINT, COPY, SCAN - UP TO 4509B061AA	274 Monoprice 104234 MPI Dell Color Laser 3010CN - Black with Chip	3 Dell Ultrabook XPS 12 Compatible Laptop Power DC Adapter Car Charger	658 ProCurve Switch 4208vl U.S ProCurve Networking 6H - J8773A#ABA	437 Dell PowerEdge R710 - 1 x X5650 - 16GB - 5 x 600GB 10K	248"
catalog_of_products=[]
# train=product_data
sampled_data,sample_test=train_test_split(product_data,train_size=0.5)
train,test=train_test_split(sampled_data,train_size=0.1)
count=0
expected_brands=train['brand_id']
actual_brands=[]
already_seen_brands=[]
# for i in range(len(expected_brands)):
# 	current_brand_id=train.iloc[i]['brand_id']
# 	out=run_rules(train.iloc[i]['product_title'])
# 	print("Title:{t}:Count{a} output{o}".format(t=train.iloc[i]['product_title'],a=count,o=out))
# 	actual_brands.append(out)
# 	count=count+1
# 	# print(count)


# sumv=0
# for values in actual_brands:
# 	if values in expected_brands:
# 		sumv=sumv+1;
	

# accuracy1=sumv/float(len(actual_brands))

# print(accuracy1)
# accuracy=np.where(actual_brands==expected_brands,1,0).sum()/float(len(actual_brands))

# print("Accuracy: {accuracy}".format(accuracy=accuracy))

print(run_rules("sandisk okk"))    	
# # print(output.ORGANIZATION)
# print(catalog_of_products)
sys.exit()


