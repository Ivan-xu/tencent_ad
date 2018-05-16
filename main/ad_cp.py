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

sysmode =['windows','ubuntu'][0]
readmode =['part','whole'][0]
params_flag =True
### mode ubuntu 
#sysmode =['windows','ubuntu'][1]
#readmode =['part','whole'][1]
#params_flag =False

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
    
    raw_user_feature=pd.read_csv(path_user_feature)
    timespent('userFeature') 
    mprint(hex(id(raw_user_feature)),'raw_user_feature')
#    raw_user_feature=[]
#    cnt=0
#    for df_user_feature in pd.read_csv(path_user_feature,chunksize=200000):
#        try:
#            
#        	df_user_feature[df_user_feature.select_dtypes(['object']).columns] = df_user_feature.select_dtypes(['object']).apply(lambda x: x.astype('category').cat.add_categories(['-1']).fill('-1'))
#        	mprint (mem_usage(df_user_feature),'mem_usage(df_user_feature)','category')
#        
#        	df_user_feature[df_user_feature.select_dtypes(['float']).columns] = df_user_feature.select_dtypes(['float']).apply(pd.to_numeric,downcast='float')
#        
#        	mprint (mem_usage(df_user_feature),'mem_usage(df_user_feature)','float')
#        	#df_user_feature[df_user_feature.select_dtypes(['int']).columns] = df_user_feature.select_dtypes(['int']).apply(pd.to_numeric,downcast='int')
#        	mprint (mem_usage(df_user_feature),'mem_usage(df_user_feature)','int')
#        except:
#        	pass
#        if cnt==0:
#            raw_user_feature =df_user_feature
#        else:
#            raw_user_feature= pd.concat([raw_user_feature,df_user_feature])
#            del df_user_feature
#            gc.collect()
#        cnt=cnt+1
#        mprint('chunk %d done.' %cnt)   


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
                    raw_user_feature = pd.DataFrame(userFeature_data) 
                   
                    userFeature_data=[]
                    raw_user_feature.to_csv(path_user_feature,index=False, header=headerflag,mode='a')   
                    headerflag =False

    #剩下的处理
        print('lastchunk done!')    
        raw_user_feature = pd.DataFrame(userFeature_data)   
        raw_user_feature.to_csv(path_user_feature, header=False,index=False,mode='a')
        timespent('userFeature')   
        
    mprint(hex(id(raw_user_feature)),'raw_user_feature')
    mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')
    
    raw_user_feature[raw_user_feature.select_dtypes(['object']).columns] = raw_user_feature.select_dtypes(['object']).apply(lambda x: x.astype('category').cat.add_categories(['-1']))
    mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')
    
    raw_user_feature[raw_user_feature.select_dtypes(['float']).columns] = raw_user_feature.select_dtypes(['float']).apply(pd.to_numeric,downcast='float')
    
    mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')
    #raw_user_feature[raw_user_feature.select_dtypes(['int']).columns] = raw_user_feature.select_dtypes(['int']).apply(pd.to_numeric,downcast='int')
    mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')


timespent('raw_user_feature') 
mprint(raw_user_feature.dtypes,'raw_user_feature.dtypes')
mprint(hex(id(raw_user_feature)),'raw_user_feature')
mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')   
user_feature = pd.DataFrame()
##start to opt the memory
for col in raw_user_feature.columns:
    dtype = raw_user_feature[col].dtypes
    mprint(type(dtype),'type dtype')
    mprint(col,'feature is :')    
    mprint(dtype,'raw_user_feature.dtype')
    mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)_before')   

#    if  dtype== np.dtype('float64'):
#        try:
#            user_feature.loc[:,col] = raw_user_feature[col].apply(pd.to_numeric,downcast='float')
#        except:
#            user_feature.loc[:,col] = raw_user_feature[col]
    if dtype== np.dtype('object'):
        num_unique_values = len(raw_user_feature[col].unique())
        num_total_values = len(raw_user_feature[col])
        if num_unique_values / num_total_values < 0.5:
#            try:    
                user_feature.loc[:,col] = raw_user_feature[col].astype('category').cat.add_categories(['-1']).fillna('-1')
                mprint(col+' as  category!')
#            except:
#                mprint('as category failed')
#                user_feature.loc[:,col] = raw_user_feature[col]
        else:
            user_feature.loc[:,col] = raw_user_feature[col]
    else:
        user_feature.loc[:,col] = raw_user_feature[col]
    ##drop the column
    user_feature.fillna('-1')
    raw_user_feature=raw_user_feature.drop(col,axis=1)
    mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)_after')   
    mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
            
mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
mail('userFeature is done!') 


ad_feature=pd.read_csv(path_ad_feature)
try:
    ad_feature[ad_feature.select_dtypes(['float']).columns] = ad_feature.select_dtypes(['float']).apply(pd.to_numeric,downcast='float')
except:
    pass


mprint(hex(id(ad_feature)),'ad_feature')
      
Chunksize =50000
readnum = 100000
train_data=pd.DataFrame()
predict_data=pd.DataFrame()
if readmode =='part':
    ##  raad train_data    
    cnt=0
    for df_train in pd.read_csv(open(path_train_csv,'r'),
                                chunksize =Chunksize,nrows=readnum):
        df_train.loc[df_train['label']==-1,'label']=0
#        try:
#            mprint (mem_usage(df_train),'mem_usage(df_train),before')
#
#            df_train[df_train.select_dtypes(['float']).columns] = df_train.select_dtypes(['float']).apply(pd.to_numeric,downcast='float')
#    
#            mprint (mem_usage(df_train),'mem_usage(df_train),after_float')
#            [df_train.select_dtypes(['int']).columns] = df_train.select_dtypes(['int']).apply(pd.to_numeric,downcast='int')
#            mprint (mem_usage(df_train),'mem_usage(df_train),after_int')
#        except :
#            pass
        df_data = pd.merge(df_train,ad_feature,on='aid',how='left')
        df_data =pd.merge(df_data,user_feature,on='uid',how='left')
        if cnt==0:
            train_data = df_data
            mprint(hex(id(train_data)),'train_data')
        else:
            train_data = pd.concat([train_data,df_data])
            mprint(hex(id(train_data)),'train_data')
        gc.collect()

        cnt=cnt+1
        mprint('chunk %d done.' %cnt)       
    timespent('read_train_data')
    
    ## read predictdata ,the same as online data
    cnt=0
    for df_predict in pd.read_csv(open(path_test1_csv,'r'),
                                chunksize =Chunksize,nrows=readnum):
#        try:
#            mprint (mem_usage(df_predict),'mem_usage(df_predict),before')
#
#            df_predict[df_predict.select_dtypes(['float']).columns] = df_predict.select_dtypes(['float']).apply(pd.to_numeric,downcast='float')
#    
#            mprint (mem_usage(df_predict),'mem_usage(df_predict),after_float')
#            [df_predict.select_dtypes(['int']).columns] = df_predict.select_dtypes(['int']).apply(pd.to_numeric,downcast='int')
#            mprint (mem_usage(df_predict),'mem_usage(df_predict),after_int')
#        except :
#            pass
        df_predict['label']=-1 
        df_data = pd.merge(df_predict,ad_feature,on='aid',how='left')
        df_data =pd.merge(df_data,user_feature,on='uid',how='left')
        if cnt==0:
            predict_data = df_data
        else:
            predict_data = pd.concat([predict_data,df_data])
        cnt=cnt+1    
    
        mprint('chunk %d done.' %cnt)     
#    data= pd.concat([train_data,predict_data])
#    mprint('data merged!')
#    mail('data merged!')


       

else:
    ##  raad train_data   
##0515 修改为一次性读取
    cnt=0
    for df_train in pd.read_csv(open(path_train_csv,'r'),
                                chunksize =Chunksize):
        df_train.loc[df_train['label']==-1,'label']=0
        df_data = pd.merge(df_train,ad_feature,on='aid',how='left')
        df_data =pd.merge(df_data,user_feature,on='uid',how='left')
        if cnt==0:
            train_data = df_data
        else:
            train_data = pd.concat([train_data,df_data])
        cnt=cnt+1
        mprint('chunk %d done.' %cnt)       
    timespent('read_train_data')
   # mail('df_train is done!') 
    
    ## read predictdata ,the same as online data
    cnt=0
    for df_predict in pd.read_csv(open(path_test1_csv,'r'),
                                chunksize =Chunksize):
        df_predict['label']=-1
        df_data = pd.merge(df_predict,ad_feature,on='aid',how='left')
        df_data =pd.merge(df_data,user_feature,on='uid',how='left')
        if cnt==0:
            predict_data = df_data
        else:
            predict_data = pd.concat([predict_data,df_data])
        cnt=cnt+1    
    
        mprint('chunk %d done.' %cnt)

data= pd.concat([train_data,predict_data])
mprint('data merged!')
mail('data merged!')
#    data.to_csv(path_newuser_feature)
#    data[data.select_dtypes(['category']).columns] = data.select_dtypes(['category']).fillna('-1')
#    data.fillna('-1')    
del train_data
del predict_data    
mprint(hex(id(data)),'data mem id')   
mprint (mem_usage(data),'mem_usage(data)_before_fill') 
data=data.dropna()  
data.fillna('-1')    

mprint(data.dtypes,'data dtypes')
#for col in data.columns:
#    mprint('fiilna'+col)
#    data[col].fillna('-1')
mprint (mem_usage(data),'mem_usage(data)_after_fill')   


mprint('start gc.collect')
gc.collect()
mprint('stop gc.collect')

one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))

    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
#for feature in one_hot_feature:
##    try:
#        data[feature] = data[feature].factorize()
#        mprint(data[feature])
