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
        mprint('%s one hot failed !'%(feature))
#for feature in one_hot_feature:
##    try:
#        data[feature] = data[feature].factorize()
#        mprint(data[feature])





def onehot_n_countvec_trans(sample):
    sample_x=sample[['creativeSize']]
    enc = OneHotEncoder()

    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        mprint(enc.n_values_,'enc.n_values_')
        sample_a=enc.transform(sample[feature].values.reshape(-1, 1))
    
        sample_x= sparse.hstack((sample_x, sample_a))
        del sample_a
#        gc.collect()
    mprint('one-hot prepared !',feature)
    #mail('onehot_trans is done!')
    
    cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    for feature in vector_feature:
        cv.fit(data[feature])
        sample_a = cv.transform(sample[feature])
        sample_x = sparse.hstack((sample_x, sample_a))
        del sample_a
#        gc.collect()
    mprint('cv prepared !','feature')
    #mail('countvec_trans is done!')
    return sample_x

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


##训练集包含正负样本
## 线上测试集
test=data[data.label==-1]
test=test.drop('label',axis=1)
res=test[['aid','uid']]
mprint(hex(id(data)),'data')

##  训练集
train=data[data.label!=-1]
train_y=train.pop('label')

##删除完整集
#del data
# 训练集、线下测试集
train, test_off, train_y, test_off_y = train_test_split(train,train_y,test_size=0.15, random_state=2018)
gc.collect()
##训练集、验证集
train, valid, train_y, valid_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
train_x = onehot_n_countvec_trans(train)
valid_x = onehot_n_countvec_trans(valid)
test_x = onehot_n_countvec_trans(test)
test_off_x =onehot_n_countvec_trans(test_off)
gc.collect()

#model=LGB_predict(train_x,train_y,test_x,res)

if params_flag ==False:
    ### 数据转换
    print('数据转换')
    lgb_train = lgb.Dataset(train_x, train_y, free_raw_data=False)
    lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train,free_raw_data=False)
    
    
    ### 设置初始参数--不含交叉验证参数
    print('设置参数')
    params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric': 'auc',
    #          'max_depth':-1,
    #          'min_data_in_leaf':20,
    #          'feature_fraction':1.0,
              }
    
    ### 交叉验证(调参)
    print('交叉验证')
    min_merror = float('-Inf')
    best_params = {}
    # 准确率
    print("调参1：提高准确率")
    for num_leaves in range(20,200,5):
        for max_depth in range(3,8,1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
    
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=2018,
                                nfold=3,
                                metrics=['auc'],
                                early_stopping_rounds=10,
                                verbose_eval=True
                                )
                
            mean_merror = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
                
            if mean_merror > min_merror:
                min_merror = mean_merror
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
                mprint(mean_merror,'mean_merror_step1')  
    params['num_leaves'] = best_params['num_leaves']
    params['max_depth'] = best_params['max_depth']
    #'''
    
    # 过拟合
    print("调参2：降低过拟合")
    for max_bin in range(1,255,5):
        for min_data_in_leaf in range(10,200,5):
#    for max_bin in range(1,255,5):
#        for min_data_in_leaf in range(10,200,20):        
                params['max_bin'] = max_bin
                params['min_data_in_leaf'] = min_data_in_leaf
                
                cv_results = lgb.cv(
                                    params,
                                    lgb_train,
                                    seed=42,
                                    nfold=3,
                                    metrics=['auc'],
                                    early_stopping_rounds=3,
                                    verbose_eval=True
                                    )
                        
                mean_merror = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
    
                if mean_merror > min_merror:
                    min_merror = mean_merror
                    best_params['max_bin']= max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
                    mprint(mean_merror,'mean_merror_step2')  
    params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    params['max_bin'] = best_params['max_bin']
    
    print("调参3：降低过拟合")
    for feature_fraction in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        for bagging_fraction in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            for bagging_freq in range(0,50,5):
#    for feature_fraction in [0.1,0.3,0.6,0.8,1.0]:
#        for bagging_fraction in [0.1,0.3,0.6,0.8,1.0]:
#            for bagging_freq in range(0,50,10):            
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq
                
                cv_results = lgb.cv(
                                    params,
                                    lgb_train,
                                    seed=42,
                                    nfold=3,
                                    metrics=['auc'],
                                    early_stopping_rounds=3,
                                    verbose_eval=True
                                    )
                        
                mean_merror = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
    
                if mean_merror > min_merror:
                    min_merror = mean_merror
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq
                    mprint(mean_merror,'mean_merror_step3')              
    params['feature_fraction'] = best_params['feature_fraction']
    params['bagging_fraction'] = best_params['bagging_fraction']
    params['bagging_freq'] = best_params['bagging_freq']
    
    print("调参4：降低过拟合")
    for lambda_l1 in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        for lambda_l2 in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            for min_split_gain in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#    for lambda_l1 in [0.1,0.3,0.6,0.8,1.0]:
#        for lambda_l2 in [0.1,0.3,0.6,0.8,1.0]:
#            for min_split_gain in [0.1,0.3,0.6,0.8,1.0]:
                params['lambda_l1'] = lambda_l1
                params['lambda_l2'] = lambda_l2
                params['min_split_gain'] = min_split_gain
                
                cv_results = lgb.cv(
                                    params,
                                    lgb_train,
                                    seed=42,
                                    nfold=3,
                                    metrics=['auc'],
                                    early_stopping_rounds=3,
                                    verbose_eval=True
                                    )
                        
                mean_merror = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
    
                if mean_merror > min_merror:
                    min_merror = mean_merror
                    best_params['lambda_l1'] = lambda_l1
                    best_params['lambda_l2'] = lambda_l2
                    best_params['min_split_gain'] = min_split_gain
                    mprint(mean_merror,'mean_merror_step4')  
    params['lambda_l1'] = best_params['lambda_l1']
    params['lambda_l2'] = best_params['lambda_l2']
    params['min_split_gain'] = best_params['min_split_gain']


    mprint(params,'best params')
    gc.collect()
else:
##params
    params ={'max_depth': 6, 'min_split_gain': 0.1, 'verbose': 1, 'lambda_l2': 0.1, 'num_leaves': 40, 'feature_fraction': 0.1, 'objective': 'binary', 'max_bin': 1, 'boosting_type': 'gbdt', 'min_data_in_leaf': 30, 'bagging_fraction': 0.8, 'bagging_freq': 10, 'lambda_l1': 0.1, 'metric': ['auc']}

def my_LGB_test(train_x,train_y,test_x,test_y):
    from multiprocessing import cpu_count
    print("LGB test")
    clf = lgb.LGBMClassifier(    
        boosting_type='gbdt', num_leaves=params['num_leaves'], reg_alpha=params['lambda_l1'], reg_lambda=params['lambda_l2'],
        max_depth=params['max_depth'], n_estimators=100, objective='binary',minmin_gain_to_split=params['min_split_gain'],
        subsample=params['bagging_fraction'], colsample_bytree=params['feature_fraction'], subsample_freq=params['bagging_freq'],
        min_data_in_leaf=params['min_data_in_leaf'],
        learning_rate=0.05,random_state=2018,n_jobs=-1
    )
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=30)
    mprint(clf.n_features_,'n_features_')
    mprint (clf.best_score_[ 'valid_0']['auc'],'clf.best_score_')
    mprint (clf.classes_,'clf.classes_')
    mprint (clf.best_iteration_,'clf.best_iteration_')
    return clf

def my_LGB_predict(train_x,train_y,valid_x,valid_y,test_x,res):
    print("LGB predict")
    from multiprocessing import cpu_count

    clf = lgb.LGBMClassifier(    
        boosting_type='gbdt', num_leaves=params['num_leaves'], reg_alpha=params['lambda_l1'], reg_lambda=params['lambda_l2'],
        max_depth=params['max_depth'], n_estimators=100, objective='binary',minmin_gain_to_split=params['min_split_gain'],
        subsample=params['bagging_fraction'], colsample_bytree=params['feature_fraction'], subsample_freq=params['bagging_freq'],
        min_data_in_leaf=params['min_data_in_leaf'],
        learning_rate=0.05,random_state=2018,n_jobs=-1
    )

    clf.fit(train_x, train_y,eval_set=[(valid_x, valid_y)],eval_metric='auc',early_stopping_rounds=30)
    ##  print the fit result
    mprint(clf.n_features_,'n_features_')
    mprint (clf.best_score_[ 'valid_0']['auc'],'clf.best_score_')
    mprint (clf.classes_,'clf.classes_')
    mprint (clf.best_iteration_,'clf.best_iteration_')
        
    
    res['score'] = clf.predict_proba(test_x,num_iteration =clf.best_iteration_)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(path_submit, index=False)
    try:
        os.system('zip baseline.zip %s'%(path_submit))
    except :
        mprint('zip baseline failed!')
    return clf



model_test = my_LGB_test(train_x,train_y,test_off_x,test_off_y)
gc.collect()
model_predict = my_LGB_predict(train_x,train_y,train_x,train_y,test_x,res)
mprint('model_test is done!')
#### 第一版
#### 训练
#params['learning_rate']=0.01
#model= lgb.train(
#          params,                     # 参数字典
#          lgb_train,                  # 训练集
#          valid_sets=lgb_eval,        # 验证集
#          num_boost_round=2000,       # 迭代次数
#          early_stopping_rounds=50    # 早停次数
#          )
#print(type(model))
#print ('线下预测')
#
#
#mprint (model.current_iteration,'current_iteration')
#preds_test_off_test =(model.predict(test_off_x,num_iteration = -1))
#preds_test_off =(model.predict(test_off_x,num_iteration = -1)>0.50).astype(int)
#
#res_testoff = [preds_test_off,test_off_y]
#print ('f1_score',metrics.f1_score(test_off_y,preds_test_off))
#print ('auc',metrics.auc(test_off_y,preds_test_off))
#preds_test_off =(model.predict(test_off_x,num_iteration = -1)> 0.50).astype(int)
#
#res_testoff = [preds_test_off,test_off_y]
#print ('f1_score',metrics.f1_score(test_off_y,preds_test_off))
#print ('auc',metrics.auc(test_off_y,preds_test_off))
#### 线上预测
#print("线上预测")
#res['score']=  model.predict_proba(test_x, num_iteration=-1) [:,1]
#res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
#res.to_csv(path_submit, index=False)

print(' - PY131 - ')

mail('ad is done!')

