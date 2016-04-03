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
from sklearn.naive_bayes import GaussianNB
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

def find_odd_brands():
	# This data set is built using the R script attached to the Repo
	c={'cid':42835
	,'0':34209
	,'1':1
	,'10':34327
	,'101':23438
	,'102':6152
	,'103':37931
	,'104':30074
	,'105':28867
	,'106':6526
	,'107':13980
	,'108':34209
	,'109':4439
	,'11':42493
	,'110':6526
	,'111':37931
	,'112':25075
	,'114':17116
	,'115':10476
	,'116':16816
	,'117':41816
	,'119':9730
	,'120':16064
	,'121':30432
	,'122':25075
	,'123':12383
	,'124':41401
	,'125':23956
	,'126':30074
	,'128':36675
	,'129':1257
	,'13':7384
	,'130':16064
	,'131':6526
	,'132':28867
	,'134':28052
	,'135':19751
	,'136':35425
	,'137':334
	,'138':16064
	,'139':30432
	,'140':6526
	,'141':5723
	,'142':42690
	,'143':5723
	,'144':16064
	,'145':6329
	,'146':6526
	,'147':16064
	,'148':30432
	,'15':25075
	,'150':15183
	,'152':5723
	,'153':6526
	,'154':6526
	,'155':12383
	,'156':27532
	,'158':44423
	,'159':36820
	,'16':12271
	,'160':16064
	,'161':16064
	,'162':25075
	,'163':30432
	,'164':17093
	,'165':30432
	,'166':37013
	,'167':32352
	,'168':6584
	,'169':14549
	,'17':34709
	,'171':30432
	,'172':23271
	,'173':45087
	,'174':36820
	,'175':19446
	,'176':8965
	,'177':6526
	,'178':18294
	,'179':9730
	,'180':32352
	,'181':12383
	,'182':5723
	,'183':42835
	,'184':6526
	,'185':28854
	,'187':5928
	,'188':42503
	,'19':16780
	,'190':28720
	,'191':34327
	,'192':5723
	,'194':13272
	,'195':16780
	,'196':21576
	,'197':42835
	,'198':34209
	,'199':25528
	,'2':42383
	,'20':22765
	,'202':25075
	,'203':34669
	,'204':25075
	,'205':6584
	,'206':42943
	,'207':30729
	,'208':34818
	,'209':6526
	,'21':35425
	,'210':43764
	,'211':6526
	,'212':18335
	,'213':42690
	,'214':34327
	,'215':9381
	,'216':5928
	,'218':5723
	,'219':11375
	,'22':12383
	,'220':12383
	,'221':34209
	,'223':1705
	,'225':42690
	,'226':42835
	,'227':5928
	,'228':25302
	,'229':6526
	,'23':8787
	,'230':6526
	,'231':12383
	,'232':44518
	,'233':32089
	,'234':34209
	,'235':30847
	,'236':2179
	,'237':7071
	,'238':27782
	,'239':37931
	,'240':1705
	,'241':4137
	,'242':42690
	,'243':16064
	,'245':22765
	,'248':42835
	,'249':34209
	,'250':1498
	,'251':21060
	,'252':14661
	,'253':6526
	,'254':6526
	,'255':2820
	,'256':43042
	,'257':43042
	,'259':25075
	,'26':34209
	,'260':45087
	,'261':12383
	,'262':33613
	,'263':42835
	,'265':12383
	,'266':13314
	,'267':25075
	,'268':1705
	,'269':13325
	,'270':6659
	,'271':42835
	,'274':42835
	,'275':37931
	,'276':42690
	,'277':6526
	,'278':34209
	,'279':22765
	,'28':19709
	,'280':18811
	,'281':5723
	,'282':13272
	,'283':34256
	,'284':26690
	,'285':34300
	,'286':5723
	,'287':26992
	,'288':6526
	,'289':20313
	,'29':4090
	,'290':16064
	,'291':10798
	,'292':6526
	,'293':25201
	,'294':6526
	,'295':30074
	,'296':34209
	,'297':8015
	,'298':26508
	,'299':42835
	,'3':27004
	,'30':7215
	,'300':34327
	,'301':40780
	,'302':25075
	,'303':29299
	,'304':6526
	,'305':42587
	,'306':6526
	,'307':30432
	,'308':12383
	,'309':19709
	,'31':41548
	,'310':3950
	,'312':32607
	,'313':14198
	,'314':6526
	,'315':30074
	,'316':12383
	,'317':5723
	,'318':15183
	,'319':37931
	,'32':31552
	,'321':5723
	,'322':12025
	,'323':6526
	,'324':143
	,'325':30074
	,'326':4684
	,'327':12383
	,'328':34327
	,'329':21244
	,'33':28720
	,'330':1736
	,'331':16064
	,'332':12383
	,'333':5723
	,'335':25075
	,'336':32841
	,'337':5723
	,'338':42690
	,'339':25075
	,'34':30847
	,'340':4380
	,'341':4475
	,'342':42587
	,'343':18185
	,'344':1584
	,'347':16860
	,'348':4380
	,'350':28720
	,'351':1669
	,'352':25075
	,'353':42835
	,'354':34209
	,'355':6526
	,'356':9791
	,'357':4380
	,'358':42835
	,'359':5723
	,'36':31483
	,'360':25075
	,'362':16064
	,'363':5723
	,'364':1705
	,'365':34209
	,'366':42383
	,'367':31194
	,'368':1761
	,'369':5723
	,'37':5928
	,'371':6526
	,'372':34209
	,'373':57
	,'374':6526
	,'375':35328
	,'376':29299
	,'377':34209
	,'378':12383
	,'379':9291
	,'38':12383
	,'381':32102
	,'382':30074
	,'383':45087
	,'384':37931
	,'385':12383
	,'386':43174
	,'387':25075
	,'388':42835
	,'39':37931
	,'390':21244
	,'391':33613
	,'392':4475
	,'393':34209
	,'395':43511
	,'396':42835
	,'397':37931
	,'398':37450
	,'4':32352
	,'40':30781
	,'400':16064
	,'402':34209
	,'403':12383
	,'404':21076
	,'405':5928
	,'406':1705
	,'407':15183
	,'408':45087
	,'409':16458
	,'41':40780
	,'410':5723
	,'411':31300
	,'413':34209
	,'414':19709
	,'415':4684
	,'416':34209
	,'418':38106
	,'419':30074
	,'42':16064
	,'420':2179
	,'421':22765
	,'422':25075
	,'423':16780
	,'424':32853
	,'425':19709
	,'426':30074
	,'427':10476
	,'428':2820
	,'429':40345
	,'43':15557
	,'430':27004
	,'431':19709
	,'432':30074
	,'434':12383
	,'435':20977
	,'436':42468
	,'437':28720
	,'438':16064
	,'439':41401
	,'44':12383
	,'442':3570
	,'443':19709
	,'444':6526
	,'445':42835
	,'447':18618
	,'448':12383
	,'449':42856
	,'45':33780
	,'450':17005
	,'451':32352
	,'452':10190
	,'453':42835
	,'454':34209
	,'455':6526
	,'456':1705
	,'457':36843
	,'459':37931
	,'46':7651
	,'460':17005
	,'461':30432
	,'462':232
	,'463':42690
	,'466':29299
	,'468':16064
	,'469':3997
	,'47':11642
	,'470':8787
	,'471':30847
	,'473':6526
	,'475':25075
	,'476':37931
	,'478':34327
	,'479':4475
	,'480':43042
	,'481':34709
	,'482':41217
	,'483':21244
	,'484':2179
	,'485':34709
	,'486':30432
	,'487':2179
	,'488':8787
	,'49':10798
	,'490':25521
	,'491':37670
	,'492':21468
	,'494':39775
	,'495':29959
	,'496':4300
	,'497':41548
	,'498':1705
	,'499':6237
	,'5':12383
	,'50':39683
	,'500':34078
	,'501':27338
	,'502':22765
	,'503':6526
	,'504':45087
	,'507':6526
	,'508':18720
	,'509':6584
	,'51':30074
	,'510':12387
	,'511':5723
	,'512':16064
	,'513':37931
	,'514':16295
	,'515':26928
	,'517':22765
	,'518':30432
	,'519':10591
	,'52':29574
	,'520':42835
	,'521':25075
	,'522':20313
	,'523':7278
	,'525':37846
	,'526':12383
	,'527':6329
	,'528':6329
	,'529':25075
	,'53':37931
	,'530':1465
	,'531':12383
	,'532':40835
	,'533':1705
	,'534':30357
	,'535':12383
	,'536':16064
	,'537':42835
	,'538':6598
	,'539':36274
	,'54':25075
	,'542':30074
	,'544':37931
	,'545':25700
	,'546':837
	,'547':10591
	,'548':5723
	,'549':40076
	,'55':44961
	,'550':32352
	,'551':12383
	,'552':27439
	,'553':30048
	,'554':34709
	,'555':6526
	,'556':5723
	,'557':42835
	,'558':2820
	,'559':4137
	,'56':9069
	,'560':6329
	,'561':34209
	,'562':28653
	,'563':34327
	,'564':42690
	,'566':12262
	,'567':6526
	,'568':12383
	,'569':11544
	,'57':5928
	,'571':38106
	,'573':2820
	,'574':34209
	,'575':6526
	,'576':6526
	,'577':12383
	,'578':34209
	,'579':30432
	,'58':42835
	,'580':44518
	,'581':12383
	,'584':12383
	,'585':15557
	,'586':25302
	,'587':6526
	,'589':6526
	,'59':12673
	,'590':30503
	,'591':5928
	,'592':6526
	,'593':5723
	,'594':37275
	,'595':25075
	,'596':6237
	,'597':14985
	,'598':18182
	,'599':34327
	,'6':11544
	,'60':23558
	,'600':2179
	,'601':25075
	,'602':7651
	,'603':42383
	,'604':25075
	,'605':6526
	,'606':18548
	,'607':18573
	,'608':34209
	,'609':1705
	,'61':41548
	,'610':12383
	,'611':42835
	,'612':6526
	,'614':42835
	,'615':37931
	,'616':42835
	,'617':12383
	,'618':11580
	,'619':34209
	,'62':14990
	,'620':25201
	,'621':25075
	,'622':6584
	,'623':36441
	,'624':43523
	,'625':6584
	,'626':23558
	,'627':30074
	,'628':34209
	,'629':12897
	,'63':12383
	,'630':22948
	,'631':2179
	,'632':25075
	,'633':5928
	,'634':9992
	,'635':25075
	,'636':6526
	,'637':6526
	,'638':20697
	,'639':5723
	,'640':34209
	,'641':6584
	,'643':6526
	,'644':18573
	,'646':28720
	,'647':44832
	,'648':6526
	,'649':34209
	,'65':16064
	,'650':9601
	,'651':13325
	,'652':25075
	,'653':18235
	,'655':38106
	,'657':36274
	,'658':3855
	,'659':16064
	,'66':12383
	,'660':42690
	,'661':42835
	,'662':25075
	,'663':30432
	,'664':25075
	,'665':1961
	,'666':42835
	,'667':44518
	,'669':19709
	,'67':18156
	,'670':34209
	,'672':16064
	,'675':17005
	,'677':16064
	,'678':16064
	,'679':30633
	,'68':42835
	,'680':12383
	,'681':30074
	,'682':30432
	,'685':22765
	,'686':34327
	,'687':9138
	,'688':10476
	,'689':12383
	,'691':30074
	,'692':30470
	,'693':4380
	,'695':30074
	,'696':12383
	,'697':5723
	,'698':9138
	,'7':6584
	,'700':20381
	,'701':6526
	,'703':34209
	,'704':25003
	,'71':36274
	,'72':6526
	,'74':2954
	,'75':45087
	,'76':13388
	,'77':42690
	,'78':45087
	,'79':42835
	,'8':8329
	,'80':12383
	,'81':22765
	,'82':16064
	,'83':6526
	,'84':40780
	,'85':21013
	,'86':42690
	,'87':16064
	,'88':42690
	,'90':40780
	,'91':6526
	,'92':30430
	,'93':34209
	,'94':5723
	,'95':31103
	,'97':42690
	,'98':42690
	,'99':22374};
	
	return c

def run_rules(product_title,product_category,odd_brands):
	if(re.match("^hp-compaq",product_title,re.IGNORECASE)):
		return 4684
	elif((re.match("^HP|hewlett",product_title,re.IGNORECASE)) and (re.match("/^(?!screen)/",product_title,re.IGNORECASE))):
		return 42835
	elif(re.match("^dell",product_title,re.IGNORECASE)):
		return 42383	
	elif(re.compile("\generic$",re.IGNORECASE).search(product_title)):
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

	

																
	try:
		return int(odd_brands[str(product_category)])
	except KeyError:
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
odd_brands=find_odd_brands()
# train,test=train_test_split(train_product_data,train_size=0.001)
print("Started Processsing....")
count=0
expected_brands=train['product_title']
actual_brands=[]
already_seen_brands=[]
start_time=time()
for i in range(len(expected_brands)):
	out=run_rules(train.iloc[i]['product_title'],train.iloc[i]['category'],odd_brands)
	# print("Count{a} ".format(a=count))
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

clf=DecisionTreeClassifier(min_samples_split= 2, criterion= 'entropy', max_depth= None, min_samples_leaf= 1) #With Optimum Hyper Parameters
# clf.fit(X,Y)
# clf=GaussianNB()
# clf=KNeighborsClassifier(n_neighbors=50,weights='distance')
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
	out=run_rules(test.iloc[i]['product_title'],test.iloc[i]['category'],odd_brands)
	# print("Count{a} ".format(a=count))
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
	predictions.append(clf.predict(testX)[0])




print("Testing Completed in {0} seconds".format(time()-start_time))
	

# # accuracy=sumv/float(len(actual_brands))
# print("Completed in {n} seconds for {k} records".format(n=end_time,k=len(expected_brands)))

# # print("Accuracy: {accuracy} for these {n} records".format(accuracy=accuracy,n=len(actual_brands)))

print("Writing output to File")
with open("output_dt"+str(time())+".txt", 'w') as f:
    for s in predictions:
        f.write(str(s) + '\n')

     


sys.exit()


