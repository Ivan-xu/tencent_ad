import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import h5py
#####
#from basic_fun.sample import mprint
##20180513
from sample import mem_usage
from sample import mprint
from  sample import mail
from sample import ftp_upload
from sample import sysmode,readmode,params_flag
from datetime import datetime
#gc.set_debug(gc.DEBUG_COLLECTABLE)

import numpy as np

now = datetime.now()
now_begin = datetime.now()
def timespent(msg=''):
    global now
    now_end = datetime.now()
    delta = now_end-now
#    delta2 = now_end - now_begin
    if msg =='':
        mprint ('last code spent-times:%s'%str(delta))
#        print ('the whole program spent-times:%s'%str(delta2))
    else:
        mprint (str(msg) +'\t spent-times:%s'%str(delta))
#        print ('the whole program spent-times:%s'%str(delta2))
    now = datetime.now()
##### mode windows

path_train_cluster_class ='/root/workspace/data/train_cluster_class'
path_train_cluster ='/root/workspace/data/train_cluster'

path_data_negative_cluster ='/root/workspace/data/data_negative_cluster.csv'
path_data_postive_cluster ='/root/workspace/data/data_postive_cluster.csv'
data_negative_cluster =pd.read_csv(path_data_negative_cluster)
data_postive_cluster =pd.read_csv(path_data_postive_cluster)


len_data_negative_cluster =len(data_negative_cluster)
leb_data_postive_cluster = len(data_postive_cluster)
##  采样
n_clusters=1001
n_p_ratio = len_data_negative_cluster/leb_data_postive_cluster
mprint(n_p_ratio,'n_p_ratio')
balance_ratio=[1.0,1.1,1.2,1.5,2.0]
for iii in balance_ratio:
	frac_ratio = iii/n_p_ratio
	mprint(frac_ratio,'frac_ratio')
	data_cluster = data_postive_cluster
	classes_null =[]
	for i in range(1,n_clusters+1,1):
	    try:
	        data_negative_class_i = data_negative_cluster.loc[data_negative_cluster['class']==i]
	        data_negative_class_i = data_negative_class_i.sample(frac = frac_ratio)
	        data_cluster = pd.concat([data_cluster, data_negative_class_i])
	    except:
	        classes_null.append(i)
	        continue
	mprint(classes_null,'classes_null')
	timespent('data_negative sapmle done! frac_ratio %s n_p_ratio %s'%(str(frac_ratio),str(iii)))
	data_cluster=data_cluster.sample(frac=1)
	path_a= path_train_cluster_class+'.csv_'+str(iii)
	path_b = path_train_cluster+'.csv_'+str(iii)
	data_cluster = data_cluster[['uid','aid','label','class']]
	data_cluster.to_csv(path_a)
	timespent("train_cluster_class has been wirted as csv. n_p_ratio%s"%(str(iii)))    
	data_cluster = data_cluster.drop('class',axis = True)
	data_cluster.to_csv(path_b,index = False)
	timespent("train_cluster  has been wirted as csv.")    


