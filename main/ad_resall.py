# -*- coding: utf-8 -*-

## 不聚类
# @author:ivan
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
from sklearn import metrics
from sample import mem_usage
from sample import mprint
from  sample import mail
from sample import ftp_upload,count_done
from sample import sysmode,readmode,params_flag
#不跑参数
params_flag = True
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
    ##NP 处理过
    path_train_csv='/root/workspace/data/train.csv'
    path_test1_csv ='/root/workspace/data/test2.csv'
    path_userFeaturedata ='/root/workspace/data/userFeature.data'
    path_submit='/root/workspace/data/submission.csv'
    def_path_log_path  ='/root/workspace/data/log/ad_'
    path_newuser_feature ='/root/workspace/data/newuserFeature.csv'
    # path_nullsubmit_data='/root/workspace/data/nullsubmission.csv'
    path_data_dtypes = '/root/workspace/data/data_dtypes.txt'
    path_data_hdf5='/root/workspace/data/balance_data_prepared_2.hdf5'
    path_balance_data_csv='/root/workspace/data/data_prepared.csv'
    path_user_feature_dtypes='/root/workspace/data/userFeature_dtypes.txt'
    path_bestparams='/root/workspace/data/best_params.txt'
    path_data_tmp_csv =  '/root/workspace/data/path_data_tmp.csv'
    
    ## 用户特征读取数量
    stpcnt=25000000
    if readmode =='part':
        # path_user_feature ='/root/workspace/data/userFeature_test.csv'
        stpcnt =1000000
else:
####    PATH
    path_user_feature='C:/Users/persp/workspace/GitHub/tencent_ad/data/userFeature.csv'
    path_ad_feature ='C:/Users/persp/workspace/GitHub/tencent_ad/data/adFeature.csv'
    path_train_csv='C:/Users/persp/workspace/GitHub/tencent_ad/data/train.csv'
    path_test1_csv ='C:/Users/persp/workspace/GitHub/tencent_ad/data/test2.csv'
    path_userFeaturedata ='C:/Users/persp/workspace/GitHub/data/ad/userFeature.data'    
    path_submit='C:/Users/persp/workspace/GitHub/tencent_ad/data/submission.csv'
    def_path_log_path  ='E:/MLfile/preliminary_contest_data/log/ad_'
    path_newuser_feature ='C:/Users/persp/workspace/GitHub/tencent_ad/data/newuserFeature.csv'
    # path_nullsubmit_data='C:/Users/persp/workspace/GitHub/tencent_ad/data/nullsubmission.csv'
    path_data_dtypes = 'C:/Users/persp/workspace/GitHub/tencent_ad/data/data_dtypes.txt'
    path_data_hdf5='C:/Users/persp/workspace/GitHub/tencent_ad/data/balance_data_prepared_2.hdf5'
    path_balance_data_csv='C:/Users/persp/workspace/GitHub/tencent_ad/data/data_prepared.csv'
    path_user_feature_dtypes='C:/Users/persp/workspace/GitHub/tencent_ad/data/userFeature_dtypes.txt'
    path_bestparams ='C:/Users/persp/workspace/GitHub/tencent_ad/data/best_params.txt'
    path_data_tmp_csv =  'C:/Users/persp/workspace/GitHub/tencent_ad/data/path_data_tmp.csv'

    ## 用户特征读取数量
    stpcnt=250000
##训练数据块读取量
if readmode =='part':
    Chunksize =100000
    readnum = 500000
    test_readnum =200000    
    stpcnt =2000000
else:
    Chunksize =int(50 *10000)
##  PATH SELECTION IS END!
one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
mprint('PROGRAM IS STARTTING!')    
if os.path.exists(path_balance_data_csv):
    with open(path_data_dtypes,"r") as f:
        dtypesread =f.read()
    column_types=eval(dtypesread)
    mprint(column_types,'data_merged column type read')    
    if readmode =='part':
        data = pd.read_csv(path_balance_data_csv,dtype =column_types,nrows =1000000)
        data.drop(data.columns[[0]],axis=1,inplace =True)
        mprint(data.shape,'data.shape')        
        timespent('data_merged part read')
#
#        data.to_csv(path_data_tmp_csv)
#        timespent('data_merged tmp save')

    else:
        data = pd.read_csv(path_balance_data_csv,dtype =column_types)
        data.drop(data.columns[[0]],axis=1,inplace =True)
        mprint(data.shape,'data.shape')        
        
        timespent('data_merged read')


else:
    if  os.path.exists(path_user_feature) and os.path.exists(path_user_feature_dtypes):
        with open(path_user_feature_dtypes,"r") as f:
            dtypesread =f.read()
        column_types=eval(dtypesread)
        mprint(column_types,'user_feature column_types read')
        #d读取用户特征数据
        if readmode =='part':
    	    user_feature=pd.read_csv(path_user_feature,dtype=column_types,nrows=stpcnt)
        else:
        	user_feature=pd.read_csv(path_user_feature,dtype=column_types)
        len_user_feature_ori = len(user_feature)
        mprint(len_user_feature_ori,'len_user_feature')        
        mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
        mprint(user_feature.dtypes,'user_feature.dtypes')

        timespent('userfeature data read finished')   

    else :
        mprint('error:run the ad_pre.py first')





    ##对AD_FEATURE数据类型转换            
                

    ad_feature=pd.read_csv(path_ad_feature)
    mprint (mem_usage(ad_feature),'mem_usage(ad_feature)')   
    mprint(ad_feature.dtypes,'ad_featured.dtypes')
    timespent('ad_feature read')

    ##针对CATEGORY类型开始数据转换
    for col in user_feature.columns:
        dtype = user_feature[col].dtypes
        mprint(dtype,'feature %s dtype'%(col))
        if dtype.name== 'category':
            user_feature.loc[:,col] = user_feature[col].cat.add_categories(['-1']).fillna('-1')
            mprint('%s feature downcast as category'%(col))
    user_feature.fillna('-1')
      
    mprint(user_feature.dtypes,'user_feature.dtypes')
    # mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
    timespent('user_feature add category -1 is done!')      

    ##  开始合并训练集、测试集
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

        mprint (mem_usage(train_data),'mem_usage(train_data)')   

        timespent('train_data read finished')
        
        ## read predictdata ,the same as online data
        cnt=0
        for df_predict in pd.read_csv(open(path_test1_csv,'r',),
                                    chunksize =Chunksize,nrows=test_readnum):
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

    # label==-1 原本是负样本 先调整负样本LAEBL为0
           

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

        mprint (mem_usage(train_data),'mem_usage(train_data)')   
            
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


        mprint (mem_usage(predict_data),'mem_usage(predict_data)')   

        timespent('predict_data read finished')

          
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

    ##抽样调参
    len_data_before =len(data)
    data.sample(frac=1,replace=True)
    data.reset_index(inplace = True)
    data.drop('index',axis =1,inplace= True)
    len_data_after =len(data)
    mprint(len_data_after,'len_data_after')
    mprint(len_data_before,'len_data_before')
    gc.collect()
    mprint('gc.collect')
    mprint (mem_usage(data),'mem_usage(data)')          

    mprint(data.dtypes,'data dtypes')






    for feature in one_hot_feature:
        try:
            mprint('%s LabelEncoder apply int  '%(feature))
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))

        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
            mprint('%s LabelEncoder astype failed !'%(feature))
        count_done(feature,one_hot_feature)
    mprint('LabelEncoder finished!')

    ## DATA DTYPES SAVES

    dtypes = data.dtypes
    dtypes_col = dtypes.index
    dtypes_type = [i.name for i in dtypes.values]

    column_types = dict(zip(dtypes_col, dtypes_type))

    with open(path_data_dtypes,"w") as f:
            f.write(str(column_types))

    try:
        data.to_csv(path_balance_data_csv)
        mprint('data_LabelEncoderd to_csv finished!')
    except:
        mprint('data_LabelEncoder to_csv failed!')


#for col in data.columns:
#    dtype = data[col].dtypes
#    mprint(dtype,'feature %s dtype'%(col))
#    if dtype.name== 'category':
#        data.loc[:,col] = data[col].cat.add_categories(['-1']).fillna('-1')
#        mprint('%s feature downcast as category'%(col))
data.fillna('-1')
  
mprint(data.dtypes,'data.dtypes')
# mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
len_data_before =len(data)
data.sample(frac=1,replace=True)
data.reset_index(inplace = True)
data.drop('index',axis =1,inplace= True)
len_data_after =len(data)
mprint(len_data_after,'len_data_after')
mprint(len_data_before,'len_data_before')
gc.collect()
mprint('gc.collect')
mprint (mem_usage(data),'mem_usage(data)')          

mprint(data.dtypes,'data dtypes')
##训练集包含正负样本
##不重采样
## 预测集（线上测试集）
test=data.loc[data['label']==-1]
test=test.drop('label',axis=1)
res=test[['aid','uid']]
mprint('test data set test split finished')
##  训练集
train=data.loc[data['label']!=-1]
train_y=train.pop('label')


##删除完整集
#del data
# 训练集、线下测试集
train, test_off, train_y, test_off_y = train_test_split(train,train_y,test_size=0.15, random_state=2018)
mprint ('data set offline split finished')
# ##训练集、验证集
train, valid, train_y, valid_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
mprint('data set valid split finished')

mem_usage_data_ori =(mem_usage(data))

mem_usage_train_ori =(mem_usage(train))
mem_usage_valid_ori =(mem_usage(valid))
mem_usage_test_ori =(mem_usage(test))
mem_usage_test_off_ori =(mem_usage(test_off))
mprint((mem_usage_data_ori),'mem_usage(data) ori ')
mprint((mem_usage_train_ori),'mem_usage(train) ori ')
mprint ((mem_usage_valid_ori),'mem_usage(valid) ori ')
mprint ((mem_usage_test_ori),'mem_usage(test) ori ')
mprint ((mem_usage_test_off_ori),'mem_usage(test_off) ori ')

####    开始ONEHOT 编码和稀疏向量化
# valid_x=valid.loc[:,['creativeSize']]
# test_x=test[:,['creativeSize']]
# test_off_x=test.loc[:,['creativeSize']]
# train_x=train.loc[:,['creativeSize']]

train_x=train[['creativeSize']]
valid_x=valid[['creativeSize']]
test_x=test[['creativeSize']]
test_off_x=test_off[['creativeSize']]

mprint('onehot_trans begin')
len_one_hot_feature=len(one_hot_feature)
for feature in one_hot_feature:
    enc = OneHotEncoder()
    enc.fit(data[feature].values.reshape(-1, 1))
    mprint('feature %s onehot.fit'%(feature))    
    mprint(enc.n_values_,'feature %s onehot enc.n_values_'%(feature))

    tmp_enc=enc.transform(train[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, tmp_enc))
    
    tmp_enc=enc.transform(valid[feature].values.reshape(-1, 1))
    valid_x= sparse.hstack((valid_x, tmp_enc))  

    tmp_enc=enc.transform(test[feature].values.reshape(-1, 1))
    test_x= sparse.hstack((test_x, tmp_enc))

    tmp_enc=enc.transform(test_off[feature].values.reshape(-1, 1))
    test_off_x= sparse.hstack((test_off_x, tmp_enc))
    del tmp_enc

    data=data.drop(feature,axis=1)
    train=train.drop(feature,axis=1)
    valid=valid.drop(feature,axis=1)
    test=test.drop(feature,axis=1)
    test_off=test_off.drop(feature,axis=1)
    gc.collect()


    mprint('feature:%s one-hot finished!'%(feature))
    count_done(feature,one_hot_feature)
mprint('onehot_trans prepared !')

mprint('countvec_trans begin')

for feature in vector_feature:
    cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    cv.fit(data[feature])
    mprint('feature %s cv.fit'%(feature))
    tmp_enc=cv.transform(train[feature])
    train_x= sparse.hstack((train_x, tmp_enc))

    tmp_enc=cv.transform(valid[feature])
    valid_x= sparse.hstack((valid_x, tmp_enc))  

    tmp_enc=cv.transform(test[feature])
    test_x= sparse.hstack((test_x, tmp_enc))

    tmp_enc=cv.transform(test_off[feature])
    test_off_x= sparse.hstack((test_off_x, tmp_enc))
    gc.collect()
    del tmp_enc

    data=data.drop(feature,axis=1)
    train=train.drop(feature,axis=1)
    valid=valid.drop(feature,axis=1)
    test=test.drop(feature,axis=1)
    test_off=test_off.drop(feature,axis=1)


    count_done(feature,vector_feature)

    mprint('feature:%s CountVectorizer finished!'%(feature))

mprint('countvec_trans prepared !')

mprint((mem_usage_data_ori),'mem_usage(data) ori ')
mprint((mem_usage_train_ori),'mem_usage(train) ori ')
mprint ((mem_usage_valid_ori),'mem_usage(valid) ori ')
mprint ((mem_usage_test_ori),'mem_usage(test) ori ')
mprint ((mem_usage_test_off_ori),'mem_usage(test_off) ori ')

mprint (mem_usage(data),'mem_usage(data) after onehot_cvtrans')
mprint (mem_usage(train),'mem_usage(train) after onehot_cvtrans')
mprint (mem_usage(valid),'mem_usage(valid) after onehot_cvtrans')
mprint (mem_usage(test),'mem_usage(test) after onehot_cvtrans')
mprint (mem_usage(test_off),'mem_usage(test_off) after onehot_cvtrans')






with open (path_bestparams,'r') as f:
	params = f.read()
	params = eval(params)
	mprint(params,'best_params read')


def my_LGB_test(train_x,train_y,test_x,test_y):
#    from multiprocessing import cpu_count
    mprint("LGB test")
    clf = lgb.LGBMClassifier(    
        boosting_type='gbdt', num_leaves=params['num_leaves'], reg_alpha=params['lambda_l1'], reg_lambda=params['lambda_l2'],
        max_depth=params['max_depth'], n_estimators=100, objective='binary',min_gain_to_split=params['min_split_gain'],
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
    mprint("LGB predict")
    # clf = lgb.LGBMClassifier(    
    #     boosting_type='gbdt', num_leaves=params['num_leaves'], reg_alpha=params['lambda_l1'], reg_lambda=params['lambda_l2'],
    #     max_depth=params['max_depth'], n_estimators=100, objective='binary',minmin_gain_to_split=params['min_split_gain'],
    #     subsample=params['bagging_fraction'], colsample_bytree=params['feature_fraction'], subsample_freq=params['bagging_freq'],
    #     min_data_in_leaf=params['min_data_in_leaf'],
    #     learning_rate=0.05,random_state=2018,n_jobs=-1
    # )
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y,eval_set=[(valid_x,valid_y)],eval_metric='auc',early_stopping_rounds=30)
    ##  print the fit result
    mprint(clf.n_features_,'n_features_')
    best_score_= clf.best_score_[ 'valid_0']['auc']
    mprint (best_score_,'clf.best_score_')
    mprint (clf.classes_,'clf.classes_')
    mprint (clf.best_iteration_,'clf.best_iteration_')
        
    
    res['score'] = clf.predict_proba(test_x,num_iteration =clf.best_iteration_)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(path_submit, index=False)
    try:
        os.system('zip baseline.zip %s'%(path_submit))
    except :
        mprint('zip baseline failed!')
    try:
        date = datetime.now().strftime('%Y%m%d_%H%M')
        score =str(float('%0.6f' %(best_score_)))
        remote_path = score+str(date)+'_submission.csv'
        local_path = path_submit
        ftp_upload(remote_path,local_path)
        mprint('ftp upload result sucess')
    except:
        mprint('ftp upload failed!')
    try:
        date = datetime.now().strftime('%Y%m%d')
        date2 =datetime.now().strftime('%Y%m%d_%H')
        remote_path = 'log_ad_'+str(date2)+'.txt'
        local_path = '/root/workspace/log/ad_'+str(date)+'.txt'
        ftp_upload(remote_path,local_path)
        mprint('ftp upload log sucess')
    except:
        mprint('ftp upload log failed!')
    
    return clf


#with open(path_bestparams,"w") as f:
#        f.write(str(params))
#mprint('best_parmas is wirted into %s'%(path_bestparams))



model_test = my_LGB_test(train_x,train_y,test_off_x,test_off_y)
model_predict = my_LGB_predict(train_x,train_y,train_x,train_y,test_x,res)
# mprint('userfeature_num :%s train_num:%s test_num:%s'%(str(len_user_feature_ori),str(readnum),str(test_readnum)))
mprint('model_params is done!')


mprint('PROGRAM IS ENDDING!')    


mail('ad is all done!')

