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
	elif(re.search("mouse pad",product_title,re.IGNORECASE)):
		return 37334
	elif(re.search("fincibo",product_title,re.IGNORECASE)):
		return 18156
	elif(re.match("^westerdigital|WD",product_title,re.IGNORECASE)):
		return 8329											
		
																
	
	
	return 4229		

	# HP Rule
	# try:
	# 	if (product_title.index("generic")>=0) or (product_title.index("for")>=0):
	# 		return 6584
	# 	elif(product_title.index("HP")>=0):
	# 		return 42835
			
	
	# except ValueError:
	# 	return 42836	

		

# def align_datattypes(product_data):
# 	data_copy=product_data.copy()
# 	for row in data_copy:
# 		temp_category=row['category']
# 		row['category']=row['category'].tostr
# 	return data_copy	

product_headers=['product_title','category']


#Data Preprocessing
product_data=get_data("classification_blind_set_corrected.tsv",product_headers)

product_data['category']=np.array(product_data['category'],dtype=np.str_)
# product_data['brand_id']=np.array(product_data['brand_id'],dtype=np.str_)
# product_data=product_data.drop_duplicates()
# print("Removed the Duplicates....")

# categories=product_data['category'].unique()
# print(categories.size)
# brands=product_data['brand_id'].unique()
# print(brands.size)


# sample_text="120GB Hard Disk Drive with 3 Years Warranty for Lenovo Essential B570 Laptop Notebook HDD Computer - Certified 3 Years Warranty from Seifelden"
# sample_text="Sony VAIO VPC-CA4S1E/W 14.0"" LCD LED Screen Display Panel WXGA HD Slim	415 CANON USA IMAGECLASS D550 - MULTIFUNCTION - MONOCHROME - LASER - PRINT, COPY, SCAN - UP TO 4509B061AA	274 Monoprice 104234 MPI Dell Color Laser 3010CN - Black with Chip	3 Dell Ultrabook XPS 12 Compatible Laptop Power DC Adapter Car Charger	658 ProCurve Switch 4208vl U.S ProCurve Networking 6H - J8773A#ABA	437 Dell PowerEdge R710 - 1 x X5650 - 16GB - 5 x 600GB 10K	248"
# catalog_of_products=[]
train=product_data
sampled_data,sample_test=train_test_split(product_data,train_size=1)
train,test=train_test_split(sampled_data,train_size=0.5)
train=product_data
count=0
expected_brands=train['product_title']
actual_brands=[]
already_seen_brands=[]
start_time=time()
for i in range(len(expected_brands)):
	# current_brand_id=train.iloc[i]['brand_id']
	out=run_rules(train.iloc[i]['product_title'],train.iloc[i]['category'])
	print("Title:{t}:Count{a} output{o}".format(t=train.iloc[i]['product_title'],a=count,o=out))
	actual_brands.append(out)
	count=count+1
	# print(count)

end_time=time()-start_time
# # sumv=0
# # for values in actual_brands:
# # 	if values in expected_brands:
# # 		sumv=sumv+1;
	

# # accuracy=sumv/float(len(actual_brands))
print("Completed in {n} seconds for {k} records".format(n=end_time,k=len(expected_brands)))

# # print("Accuracy: {accuracy} for these {n} records".format(accuracy=accuracy,n=len(actual_brands)))

print("Writing output to File")
with open("output"+str(time())+".txt", 'w') as f:
    for s in actual_brands:
        f.write(str(s) + '\n')

# print(run_rules("sandisk okk"))    	
# # print(output.ORGANIZATION)
# print(catalog_of_products)
sys.exit()


