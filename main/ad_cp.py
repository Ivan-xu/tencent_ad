# -*- coding: utf-8 -*-
# @author:ivan
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




if sysmode == 'ubuntu':
####    PATH
    path_user_feature='/root/workspace/data/userFeature.csv'
    path_ad_feature ='/root/workspace/data/adFeature.csv'
    path_train_csv='/root/workspace/data/train.csv'
    path_test1_csv ='/root/workspace/data/test1.csv'
    path_userFeaturedata ='/root/workspace/data/userFeature.data'
    path_submit='/root/workspace/data/submission.csv'
    def_path_log_path  ='/root/workspace/data/log/ad_'
    path_newuser_feature ='/root/workspace/data/newuserFeature.csv'
    path_nullsubmit_data='/root/workspace/data/nullsubmission.csv'
    ## 用户特征读取数量
    stpcnt=25000000
    if readmode =='part':
        path_user_feature ='/root/workspace/data/userFeature_test.csv'
        stpcnt =1000000
else:
####    PATH
    path_user_feature='E:/MLfile/preliminary_contest_data/data/userFeature.csv'
    path_ad_feature ='E:/MLfile/preliminary_contest_data/data/adFeature.csv'
    path_train_csv='E:/MLfile/preliminary_contest_data/data/train.csv'
    path_test1_csv ='E:/MLfile/preliminary_contest_data/data/test1.csv'
    path_userFeaturedata ='C:/Users/persp/workspace/GitHub/data/ad/userFeature.data'    
    path_submit='E:/MLfile/preliminary_contest_data/data/submission.csv'
    def_path_log_path  ='E:/MLfile/preliminary_contest_data/log/ad_'
    path_newuser_feature ='E:/MLfile/preliminary_contest_data/data/newuserFeature.csv'
    path_nullsubmit_data='E:/MLfile/preliminary_contest_data/data/nullsubmission.csv'

    ## 用户特征读取数量
    stpcnt=250000
    
##  PATH SELECTION IS END!
mprint('PROGRAM IS STARTTING!')    
    
if os.path.exists(path_user_feature):
    
    raw_user_feature=pd.read_csv(path_user_feature)
    timespent('userFeature') 
    mprint(hex(id(raw_user_feature)),'raw_user_feature') 
else:
    userFeature_data = []
    headerflag=True
    cnt =0
    chunk =200000
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



timespent('read raw_user_feature finished') 
mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')   
mprint(raw_user_feature.dtypes,'raw_user_feature.dtypes')
user_feature = pd.DataFrame()
##start to opt the memory
for col in raw_user_feature.columns:
    dtype = raw_user_feature[col].dtypes
    if  dtype== np.dtype('int64'):
        try:
            user_feature.loc[:,col] = raw_user_feature[col].apply(pd.to_numeric,downcast='int')
            mprint('%s feature downcast as int'%(col))
        except:
            user_feature.loc[:,col] = raw_user_feature[col]
    if  dtype== np.dtype('float64'):
        try:
            user_feature.loc[:,col] = raw_user_feature[col].apply(pd.to_numeric,downcast='float')
            mprint('%s feature downcast as float'%(col))
        except:
            user_feature.loc[:,col] = raw_user_feature[col]
    if dtype== np.dtype('object'):
        num_unique_values = len(raw_user_feature[col].unique())
        num_total_values = len(raw_user_feature[col])
        if num_unique_values / num_total_values < 0.5:
            try:
                user_feature.loc[:,col] = raw_user_feature[col].astype('category').cat.add_categories(['-1']).fillna('-1')
                mprint('%s feature downcast as category'%(col))
            except:
                user_feature.loc[:,col] = raw_user_feature[col]
        else:
            user_feature.loc[:,col] = raw_user_feature[col]
    else:
        user_feature.loc[:,col] = raw_user_feature[col]
    ##drop the column
    user_feature.fillna('-1')
    raw_user_feature=raw_user_feature.drop(col,axis=1)
    
    #mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)_after')   
    #mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
mprint('user_feature casttype is done!')               
mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
mprint(user_feature.dtypes,'user_feature.dtypes')

            

raw_ad_feature=pd.read_csv(path_ad_feature)
ad_feature = pd.DataFrame()
mprint (mem_usage(raw_ad_feature),'mem_usage(raw_ad_feature)')   
mprint(ad_feature.dtypes,'ad_featured.dtypes')

for col in raw_ad_feature.columns:
    dtype = raw_ad_feature[col].dtypes
    if  dtype== np.dtype('int64'):
        try:
            ad_feature.loc[:,col] = raw_ad_feature[col].apply(pd.to_numeric,downcast='int')
            mprint('%s feature downcast as int'%(col))
        except:
            ad_feature.loc[:,col] = raw_ad_feature[col]
    if  dtype== np.dtype('float64'):
        try:
            ad_feature.loc[:,col] = raw_ad_feature[col].apply(pd.to_numeric,downcast='float')
            mprint('%s feature downcast as float'%(col))
        except:
            ad_feature.loc[:,col] = raw_ad_feature[col]
    if dtype== np.dtype('object'):
        num_unique_values = len(raw_ad_feature[col].unique())
        num_total_values = len(raw_ad_feature[col])
        if num_unique_values / num_total_values < 0.5:
            try:
                ad_feature.loc[:,col] = raw_ad_feature[col].astype('category').cat.add_categories(['-1']).fillna('-1')
                mprint('%s feature downcast as category'%(col))
            except:
                ad_feature.loc[:,col] = raw_ad_feature[col]
        else:
            ad_feature.loc[:,col] = raw_ad_feature[col]
    else:
        ad_feature.loc[:,col] = raw_ad_feature[col]
    ##drop the column
    ad_feature.fillna('-1')
    raw_ad_feature=raw_ad_feature.drop(col,axis=1)
    #mprint (mem_usage(raw_ad_feature),'mem_usage(raw_ad_feature)_after')   
    #mprint (mem_usage(ad_feature),'mem_usage(ad_feature)')   
            
mprint('ad_feature casttype is done!') 
mprint (mem_usage(ad_feature),'mem_usage(ad_feature)')   
mprint(ad_feature.dtypes,'ad_featured.dtypes')


      
Chunksize =500000
readnum = 200000

train_data=pd.DataFrame()
predict_data=pd.DataFrame()
if readmode =='part':
    ##  raad train_data    
    cnt=0
    for df_train in pd.read_csv(open(path_train_csv,'r'),
                                chunksize =Chunksize,nrows=readnum):
        df_train.loc[df_train['label']==-1,'label']=0
        df_data = pd.merge(df_train,ad_feature,on='aid',how='left')
        df_data =pd.merge(df_data,user_feature,on='uid',how='left')
        if cnt==0:
            train_data = df_data
            mprint(hex(id(train_data)),'train_data')
        else:
            train_data = pd.concat([train_data,df_data])
            mprint(hex(id(train_data)),'train_data')

        cnt=cnt+1
        mprint('chunk %d done.' %cnt)       
    timespent('train_data read finished')
    
    ## read predictdata ,the same as online data
    cnt=0
    for df_predict in pd.read_csv(open(path_test1_csv,'r'),
                                chunksize =Chunksize,nrows=readnum):
        df_predict['label']=-1 
        df_data = pd.merge(df_predict,ad_feature,on='aid',how='left')
        df_data =pd.merge(df_data,user_feature,on='uid',how='left')
        if cnt==0:
            predict_data = df_data
        else:
            predict_data = pd.concat([predict_data,df_data])
        cnt=cnt+1    
    
        mprint('chunk %d done.' %cnt)     
    timespent('predict_data read finished')


       

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
        del df_data
        gc.collect()
        mprint('chunk %d done.' %cnt)       
        
    timespent('train_data read finished')
    
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
        del df_data        
        gc.collect()    
        mprint('chunk %d done.' %cnt)
    timespent('predict_data read finished')


mprint (mem_usage(train_data),'mem_usage(train_data)')          
mprint (mem_usage(predict_data),'mem_usage(predict_Data)')           
mprint(train_data.dtypes,'train_data.dtypes')
mprint(predict_data.dtypes,'predict_data.dtypes')
len_train_data= len(train_data)
len_predict_data = len(predict_data)
mprint('len_train_data %d'%(len_train_data))
mprint('len_predict_data %d'%(len_predict_data))
mprint('train/predict ratio: %s'%(len_train_data/len_predict_data))
len_train_data_postive= len(train_data[train_data['label']==1])
len_train_data_negative= len(train_data[train_data['label']==0])
mprint(len_train_data_postive,'len_train_data_postive')
mprint(len_train_data_negative,'len_train_data_negative')
mprint('N/P ratio: %s'%(len_train_data_negative/len_train_data_postive))
train_data.fillna('-1',inplace= True)
predict_data.fillna('-1',inplace =True)
data= pd.concat([train_data,predict_data])
mail('data fillna and merged!')  
del train_data
del predict_data    
mprint (mem_usage(data),'mem_usage(data)')          

mprint(data.dtypes,'data dtypes')

mprint('start gc.collect')
gc.collect()
mprint('stop gc.collect')









def LGB_test(train_x,train_y,test_x,test_y):
    from multiprocessing import cpu_count
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50,random_state=2018,n_jobs=cpu_count()-1
    )
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)
    # print(clf.feature_importances_)
    return clf,clf.best_score_[ 'valid_1']['auc']

def LGB_predict(train_x,train_y,test_x,res):
    print("LGB predict")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=100
    )
#    clf.fit(train_x, train_y, eval_set=[(valid, valid_y)], eval_metric='auc',early_stopping_rounds=100)
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(valid_x,valid_y)],eval_metric='auc',early_stopping_rounds=100)
    
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(path_submit, index=False)
    os.system('zip baseline.zip %s'%(path_submit))
    return clf


one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        mprint('%s LabelEncoder apply int  '%(feature))
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))

    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
        mprint('%s LabelEncoder failed !'%(feature))
mprint('LabelEncoder finished!')


##训练集包含正负样本
## 线上测试集
test=data[data.label==-1]
test=test.drop('label',axis=1)
res=test[['aid','uid']]
mprint('data set test split finished')
##  训练集
train=data[data.label!=-1]
train_y=train.pop('label')
mprint('data set train split finished')
mprint('del data and free m')
gc.collect()

##删除完整集
#del data
# 训练集、线下测试集
train, test_off, train_y, test_off_y = train_test_split(train,train_y,test_size=0.15, random_state=2018)
mprint ('data set offline split finished')
##训练集、验证集
train, valid, train_y, valid_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
mprint('data set valid split finished')
mem_usage_data_before = mem_usage(data)
mem_usage_data_train = mem_usage(train)
mem_usage_data_valid = mem_usage(valid)
mem_usage_data_test = mem_usage(test)
mem_usage_data_test_off = mem_usage(test_off)

sample=[train,valid,test,test_off]
sample_name=['train','vaild','test','test_off']
len_samplelist = len(sample)
sample_x=[]
for i in range(len_samplelist):
    print (type(sample[i]))
    #空数组 不能越界
    sample_x.append(sample[i][['creativeSize']])

for feature in one_hot_feature:
    enc = OneHotEncoder()
    enc.fit(data[feature].values.reshape(-1, 1))
    mprint('feature:%s one_hot fit'%(feature)) 

    mprint(enc.n_values_,feature+'enc.n_values_')
    sample_a={}
    ## 每个数据集独热编码
    for i in range(len_samplelist):

        sample_a[i]=enc.transform(sample[i][feature].values.reshape(-1, 1))
    
        sample_x[i]= sparse.hstack((sample_x[i], sample_a[i]))
        del sample_a[i]
        mprint (sample_x[i],'sample_x[%s]:%s feature:%s one_hot_trans finished'%(str(i),sample_name[i],feature))
    data=data.drop(feature,axis=1)
    gc.collect()
    mprint (mem_usage(data),'mem_usage(data)')
    mprint('feature:%s one-hot finished!'%(feature))

mprint('one_hot_trans finished !')

for feature in vector_feature:
    cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    cv.fit(data[feature])
    mprint('feature:%s one_hot fit'%(feature)) 
    sample_a={}
    for i in range(len_samplelist):

        sample_a[i] = cv.transform(sample[i][feature])
        sample_x[i] = sparse.hstack((sample_x[i], sample_a[i]))
        del sample_a[i]
        mprint (sample_x[i],'sample_x[%s]:%s feature:%s countvec_trans finished'%(str(i),sample_name[i],feature))
    data = data.drop(feature,axis=1)
    gc.collect()
    mprint (mem_usage(data),'mem_usage(data)')
    mprint('feature:%s CountVectorizer finished!'%(feature))

mprint('countvec_trans finished !')
#mail('countvec_trans is done!')

train_x= sample_x[0]
valid_x= sample_x[1]
test_x =sample_x[2]
test_off_x=sample_x[3]
mprint (mem_usage(data),'mem_usage(data)')

del sample_x
mprint(train_x,'train_x')
mprint(valid_x,'valid_x')
del sample
mprint (mem_usage(train),'mem_usage(train)')
mprint (mem_usage(valid),'mem_usage(valid)')
mprint (mem_usage(test),'mem_usage(test)')
mprint (mem_usage(test_off),'mem_usage(test_off)')


mprint (mem_usage(test_x),'mem_usage(test_x)')
mprint (mem_usage(valid_x),'mem_usage(valid_x)')
mprint (mem_usage(train_x),'mem_usage(train_x)')
mprint (mem_usage(test_off_x),'mem_usage(test_off_x)')

del data
gc.collect()
#model=LGB_predict(train_x,train_y,test_x,res)