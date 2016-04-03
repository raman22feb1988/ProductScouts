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

# def find_catalog(product_name,catalog_of_products):
# 	temp_catalog=list(catalog_of_products)
# 	tagged_text=pos_tag(product_name.split())
# 	output=nltk.ne_chunk(tagged_text)
# 	for subtree in output.subtrees(filter=lambda t: t.label() == 'PERSON'):
# 		for leave in subtree.leaves():
# 			temp_catalog.append(leave[0])
# 	return temp_catalog	


def run_rules(product_title,product_category):
	if((re.match("^HP|hewlett",product_title,re.IGNORECASE)) and (re.match("/^(?!screen)/",product_title,re.IGNORECASE))):
		return 42835
	elif(re.match("^dell",product_title,re.IGNORECASE)):
		return 42383	
	elif(re.search("generic",product_title,re.IGNORECASE)):
		return 6584
	elif(((re.search("card",product_title,re.IGNORECASE)) or (re.search("adapter",product_title,re.IGNORECASE))) and (product_category==618)):
		return 19709
	elif(re.match("^decal|matte",product_title,re.IGNORECASE)):
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
	elif(re.match("^sodo",product_title,re.IGNORECASE)):
		return 21244
	elif(re.match("^hp-compaq",product_title,re.IGNORECASE)):
		return 4684
	elif((re.match("^thosiba",product_title,re.IGNORECASE)) and (re.search("card",product_title,re.IGNORECASE))):
		return 35099
	elif((re.match("^cisco",product_title,re.IGNORECASE)) and (re.match("/^(?!compatible)/",product_title,re.IGNORECASE))):
		return 28720
	elif((re.search("animal",product_title,re.IGNORECASE)) and ((re.search("customized",product_title,re.IGNORECASE)) or (re.search("designed",product_title,re.IGNORECASE)) or (re.search("cool",product_title,re.IGNORECASE)))):
		return 17005
	elif((re.search("animal",product_title,re.IGNORECASE)) and ((re.search("protection",product_title,re.IGNORECASE)) )):
		return 44466
	elif(((re.search("tie",product_title)) and (product_category==466)) or (re.search("multiple colors",product_title,re.IGNORECASE)) or (re.search("nextdia",product_title,re.IGNORECASE))):
		return 43042
	elif(re.search("belkin",product_title,re.IGNORECASE)):
		return 15557
	elif(re.search("product category",product_title,re.IGNORECASE)):
		return 42503		
	elif((re.match("^acer",product_title,re.IGNORECASE)) and (re.match("/^(?!for)/",product_title,re.IGNORECASE))):
		return 2989	
	elif(re.match("^startech",product_title,re.IGNORECASE)):
		return 4684
	elif(re.match("^tripp",product_title,re.IGNORECASE)):
		return 13325
	elif(re.match("^lenovo",product_title,re.IGNORECASE)):
		return 37923
	elif(re.match("^3drose",product_title,re.IGNORECASE)):
		return 45087
	elif(re.search("superb choice",product_title,re.IGNORECASE)):
		return 34996
	elif(re.match("^samsung",product_title,re.IGNORECASE)):
		return 9381
	elif(re.match("^asus",product_title,re.IGNORECASE)):
		return 33613
	elif(re.match("^lexmark",product_title,re.IGNORECASE)):
		return 13357
	elif(re.match("^intel",product_title,re.IGNORECASE)):
		return 21076
	elif(re.search("Seifelden",product_title,re.IGNORECASE)):
		return 3950
	elif(re.search("mouse pad computer mousepad",product_title,re.IGNORECASE)):
		return 37334
	elif(re.search("mouse pad",product_title,re.IGNORECASE)):
		return 41873	
	elif(re.match("^sony vaio",product_title,re.IGNORECASE)):
		return 35585
	elif(re.search("replacement laptop keyboard",product_title,re.IGNORECASE)):
		return 42587
	elif(re.match("^canon",product_title,re.IGNORECASE)):
		return 43043
	elif(re.match("^epson",product_title,re.IGNORECASE)):
		return 28653
	elif(re.search("roocase",product_title,re.IGNORECASE)):
		return 4380
	elif(re.match("^xerox",product_title,re.IGNORECASE)):
		return 36778
	elif(re.search("design protective decal skin sticker",product_title,re.IGNORECASE)):
		return 1498
	elif((re.match("^FORD",product_title,re.IGNORECASE)) and (re.search("computer module ECM ECU",product_title,re.IGNORECASE))) :
		return 3570
	elif(re.search("skinit skin",product_title,re.IGNORECASE)):
		return 17116
	elif(re.search("crest coat",product_title,re.IGNORECASE)):
		return 20607
	elif(re.search("mightyskins",product_title,re.IGNORECASE)):
		return 3406
	elif(re.search("boxwave",product_title,re.IGNORECASE)):
		return 18573
	elif(re.search("cellet",product_title,re.IGNORECASE)):
		return 23480
	elif(re.search("fincibo",product_title,re.IGNORECASE)):
		return 18156
	elif(re.match("^westerdigital|WD",product_title,re.IGNORECASE)):
		return 8329
	elif((re.match("^apple",product_title,re.IGNORECASE)) and ((re.match("/^(?!by)/",product_title,re.IGNORECASE)) or (re.match("/^(?!for)/",product_title,re.IGNORECASE)))):
		return 36043
	elif((re.search("skinguardz",product_title,re.IGNORECASE))):
		return 22028

		
																
	
	
	return 42835		



training_product_headers=['product_title','brand_id','category']
training_filename="classification_train.tsv"

testing_product_headers=['product_title','category']
testing_filename="classification_blind_set_corrected.tsv"

#Data Preprocessing
train_product_data=get_data(training_filename,training_product_headers)


train_product_data['brand_id']=np.array(train_product_data['brand_id'],dtype=np.int32)
train_product_data=train_product_data.drop_duplicates()
train_product_data['category']=np.array(train_product_data['category'],dtype=np.int32)

train=train_product_data
# train,test=train_test_split(train_product_data,train_size=0.001)
count=0
expected_brands=train['product_title']
actual_brands=[]
already_seen_brands=[]
start_time=time()
for i in range(len(expected_brands)):
	out=run_rules(train.iloc[i]['product_title'],train.iloc[i]['category'])
	print("Count{a} ".format(a=count))
	# train['boosting_feature'].append(out)
	actual_brands.append(out)
	count=count+1
	
train['boosting_feature']=actual_brands
end_time=time()-start_time



#Classification Algorithm
train_features=[]
train_features.append(train.columns[2])
train_features.append(train.columns[3])

X=train[train_features]
Y=train['brand_id']

clf=DecisionTreeClassifier(min_samples_split= 2, max_leaf_nodes= 60, criterion= 'entropy', max_depth= None, min_samples_leaf= 1) #With Optimum Hyper Parameters
clf.fit(X,Y)
print("Training Completed in {0} seconds".format(time()-start_time))

test_product_data=get_data(testing_filename,testing_product_headers)
test_product_data['category']=np.array(test_product_data['category'],dtype=np.str_)
test=test_product_data
# test,test2=train_test_split(test_product_data,train_size=0.001)
expected_brands=test['product_title']
test_actual_brands=[]
already_seen_brands=[]
start_time=time()
count=0
for i in range(len(expected_brands)):
	out=run_rules(test.iloc[i]['product_title'],test.iloc[i]['category'])
	print("Count{a} ".format(a=count))
	# test['boosting_feature'].append(out)
	test_actual_brands.append(out)
	count=count+1

test['boosting_feature']=test_actual_brands

test_features=[]
test_features.append(test.columns[1])
test_features.append(test.columns[2])
testX=test[test_features]

print("Testing Started...")
start_time=time()
predictions=[]
for i in range(len(expected_brands)):
	testX=test.iloc[i][test_features]
	predictions.append(clf.predict(testX))




print("Testing Completed in {0} seconds".format(time()-start_time))
	

# # accuracy=sumv/float(len(actual_brands))
# print("Completed in {n} seconds for {k} records".format(n=end_time,k=len(expected_brands)))

# # print("Accuracy: {accuracy} for these {n} records".format(accuracy=accuracy,n=len(actual_brands)))

print("Writing output to File")
with open("output_dt"+str(time())+".txt", 'w') as f:
    for s in predictions:
        f.write(str(s) + '\n')

print("Writing output to File")
with open("output_normal"+str(time())+".txt", 'w') as f:
    for s in test_actual_brands:
        f.write(str(s) + '\n')        

# print(run_rules("sandisk okk"))    	
# # print(output.ORGANIZATION)
# print(catalog_of_products)
sys.exit()


