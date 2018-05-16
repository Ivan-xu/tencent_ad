# -*- coding: utf-8 -*-
# @author:bryan
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc

#####
#from basic_fun.sample import mprint
##20180513
from sklearn import metrics
from sample import mem_usage
from sample import mprint
from  sample import mail
from datetime import datetime
#gc.set_debug(gc.DEBUG_COLLECTABLE)


now = datetime.now()
now_begin = datetime.now()
def timespent(msg=''):
    global now
    now_end = datetime.now()
    delta = now_end-now
    delta2 = now_end - now_begin
    if msg =='':
        mprint ('last code spent-times:%s'%str(delta))
#        print ('the whole program spent-times:%s'%str(delta2))
    else:
        mprint (str(msg) +'\t spent-times:%s'%str(delta))
#        print ('the whole program spent-times:%s'%str(delta2))
    now = datetime.now()
##### mode windows

#sysmode =['windows','ubuntu'][0]
#readmode =['part','whole'][0]
#params_flag =False
### mode ubuntu 
sysmode =['windows','ubuntu'][0]
readmode =['part','whole'][0]
params_flag =True

if sysmode == 'ubuntu':
####    PATH
    path_user_feature='/root/workspace/data/userFeature.csv'
    path_user_feature_pre='/root/workspace/data/userFeature_'
    path_ad_feature ='/root/workspace/data/adFeature.csv'
    path_train_csv='/root/workspace/data/train.csv'
    path_test1_csv ='/root/workspace/data/test1.csv'
    path_userFeaturedata ='/root/workspace/data/userFeature.data'
    path_submit='/root/workspace/data/submission.csv'
    def_path_log_path  ='/root/workspace/data/log/ad_'
    path_newuser_feature ='/root/workspace/data/newuserFeature.csv'
    ## 用户特征读取数量
    stpcnt=25000000

else:
####    PATH
    path_user_feature='E:/MLfile/preliminary_contest_data/data/userFeature.csv'
    path_user_feature_pre='E:/MLfile/preliminary_contest_data/data/userFeature_'
    path_ad_feature ='E:/MLfile/preliminary_contest_data/data/adFeature.csv'
    path_train_csv='E:/MLfile/preliminary_contest_data/data/train.csv'
    path_test1_csv ='E:/MLfile/preliminary_contest_data/data/test1.csv'
    path_userFeaturedata ='C:/Users/persp/workspace/GitHub/data/ad/userFeature.data'    
    path_submit='E:/MLfile/preliminary_contest_data/data/submission.csv'
    def_path_log_path  ='E:/MLfile/preliminary_contest_data/log/ad_'
    path_newuser_feature ='E:/MLfile/preliminary_contest_data/data/newuserFeature.csv'
    ## 用户特征读取数量
    stpcnt=250000
    
##  PATH SELECTION IS END!
    
    
if os.path.exists(path_user_feature):
    user_feature=pd.read_csv(path_user_feature)
    timespent('userFeature') 
    mprint(hex(id(user_feature)),'user_feature')
else:
    userFeature_data = []
    headerflag=True
    cnt =0
    chunk =100000
#    stpcnt =20000000
    cnt_i=0
    with open(path_userFeaturedata, 'r') as f:
        for i, line in enumerate(f):
            if i==stpcnt:
                break
            cnt_i=  cnt_i+1
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)        
            if (i+1) % chunk == 0:
                cnt=cnt+1
                print (i+1)
                print('chunk %d done.' %cnt)   
                if stpcnt-(i+1)<=chunk:
                    print ('lastchunk')
                    continue
                else:
                    user_feature = pd.DataFrame(userFeature_data) 
                    userFeature_data=[]

                    user_feature.to_csv(path_user_feature,index=False, header=headerflag,mode='a')   
                    headerflag =False

    #剩下的处理
        print('lastchunk done!')    
        user_feature = pd.DataFrame(userFeature_data)   
        user_feature.to_csv(path_user_feature, header=False,index=False,mode='a')
        timespent('userFeature')   
mprint (mem_usage(user_feature),'mem_usage(user_feature)')

#user_feature[user_feature.select_dtypes(['object']).columns] = user_feature.select_dtypes(['object']).apply(lambda x: x.astype('category'))
#object_feature = user_feature.select_dtypes(include=['category'])
dtypes = user_feature.dtypes

dtypes_col = dtypes.index
dtypes_type = [i.name for i in dtypes.values]

column_types = dict(zip(dtypes_col, dtypes_type))
preview = first2pairs = {key:value for key,value in list(column_types.items())[:10]}

#mprint(type(object_feature['kw1'].describe()),'type')
#
#mprint(object_feature['kw1'].describe())
#mprint (object_feature['kw1'].cat.codes)
#mprint (mem_usage(user_feature),'mem_usage(user_feature)')
#
#mprint (user_feature.describe())