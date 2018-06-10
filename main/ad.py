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
    path_train_csv='/root/workspace/data/train_cluster.csv'
    path_test1_csv ='/root/workspace/data/test2.csv'
    path_userFeaturedata ='/root/workspace/data/userFeature.data'
    path_submit='/root/workspace/data/submission.csv'
    def_path_log_path  ='/root/workspace/data/log/ad_'
    path_newuser_feature ='/root/workspace/data/newuserFeature.csv'
    path_data_dtypes = '/root/workspace/data/balance_data_dtypes_sample.txt'
    path_data_hdf5='/root/workspace/data/balance_data_prepared_2.hdf5'
    path_balance_data_merge_feature_csv='/root/workspace/data/balance_data_merge_feature.csv'
    path_user_feature_dtypes='/root/workspace/data/userFeature_dtypes.txt'
    path_bestparams='/root/workspace/data/best_params.txt'
    ## 用户特征读取数量
    stpcnt=25000000
    if readmode =='part':
        path_user_feature ='/root/workspace/data/userFeature_test.csv'
        stpcnt =250000
else:
####    PATH
    path_user_feature='C:/Users/persp/workspace/GitHub/data/userFeature.csv'
    path_ad_feature ='C:/Users/persp/workspace/GitHub/data/adFeature.csv'
    path_train_csv='C:/Users/persp/workspace/GitHub/data/train_cluster.csv'
    path_test1_csv ='C:/Users/persp/workspace/GitHub/data/test2.csv'
    path_userFeaturedata ='C:/Users/persp/workspace/GitHub/tencent_ad/data/userFeature.data'    
    path_submit='C:/Users/persp/workspace/GitHub/data/submission.csv'
    def_path_log_path  ='E:/MLfile/preliminary_contest_data/log/ad_'
    path_newuser_feature ='C:/Users/persp/workspace/GitHub/data/newuserFeature.csv'
    path_data_dtypes = 'C:/Users/persp/workspace/GitHub/data/balance_data_dtypes_sample.txt'
    path_data_hdf5='C:/Users/persp/workspace/GitHub/data/balance_data_prepared_2.hdf5'
    path_balance_data_merge_feature_csv='C:/Users/persp/workspace/GitHub/data/balance_data_merge_feature.csv'
    path_user_feature_dtypes='C:/Users/persp/workspace/GitHub/data/userFeature_dtypes.txt'
    path_bestparams ='C:/Users/persp/workspace/GitHub/data/best_params.txt'

    ## 用户特征读取数量
    stpcnt=250000
##训练数据块读取量
if readmode =='part':
    Chunksize =20000
    readnum = 20000    
else:
    Chunksize =int(50 *10000)
##  PATH SELECTION IS END!
mprint('PROGRAM IS STARTTING!')    

if  os.path.exists(path_user_feature) and os.path.exists(path_user_feature_dtypes):
    with open(path_user_feature_dtypes,"r") as f:
        dtypesread =f.read()
    column_types=eval(dtypesread)
    mprint(column_types,'user_feature column_types read')
    #d读取用户特征数据
    user_feature=pd.read_csv(path_user_feature,dtype=column_types)
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
mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
timespent('user_feature add category -1 is done!')      

##  开始合并训练集、测试集
train_data=pd.DataFrame()
predict_data=pd.DataFrame()
if readmode =='part':
    ##  raad train_data    
    cnt=0
    for df_train in pd.read_csv(open(path_train_csv,'r'),
                                chunksize =Chunksize,nrows=readnum):
        # df_train.loc[df_train['label']==-1,'label']=0
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

# label==-1 原本是负样本 先调整负样本LAEBL为0
       

else:
    ##  raad train_data   
##0515 修改为一次性读取
    cnt=0
    for df_train in pd.read_csv(open(path_train_csv,'r'),
                                chunksize =Chunksize):
        # df_train.loc[df_train['label']==-1,'label']=0
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

    mprint (mem_usage(train_data),'mem_usage(train_data)')   

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

## 保存读取数据

dtypes = data.dtypes
dtypes_col = dtypes.index
dtypes_type = [i.name for i in dtypes.values]

column_types = dict(zip(dtypes_col, dtypes_type))

with open(path_data_dtypes,"w") as f:
        f.write(str(column_types))

try:
    data.to_csv(path_balance_data_merge_feature_csv)
    mprint('path_balance_data_merge_feature_csv to_csv finished!')
except:
    mprint('path_balance_data_merge_feature_csv to_csv failed!')


##抽样调参
mprint (mem_usage(data),'mem_usage(data)')          
len_data_before =len(data)
cross_valid_ratio = 0.5
msk_1 = np.random.rand(len(data)) < cross_valid_ratio
data = data.loc[msk_1]
data.sample(frac=1,replace=True)
data.reset_index(inplace = True)
data.drop('index',axis =1,inplace= True)
mprint('data sample is ok  sample ratio is %s'%(str(cross_valid_ratio)))
len_data_after =len(data)
mprint(len_data_after,'len_data_after')
mprint(len_data_before,'len_data_before')
gc.collect()
mprint('gc.collect')
mprint (mem_usage(data),'mem_usage(data)')          

mprint(data.dtypes,'data dtypes')


one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        mprint('%s LabelEncoder apply int  '%(feature))
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))

    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
        mprint('%s LabelEncoder astype failed !'%(feature))
    count_done(feature,one_hot_feature)
mprint('LabelEncoder finished!')


##训练集包含正负样本

#len_data_after = len(data)
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
train, test_off, train_y, test_off_y = train_test_split(train,train_y,test_size=0.1, random_state=2018)
mprint ('data set offline split finished')
# ##训练集、验证集
train, valid, train_y, valid_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
mprint('data set valid split finished')

mem_usage_data_ori =(mem_usage(data))
mem_usage_train_ori =(mem_usage(train))
mem_usage_valid_ori =(mem_usage(valid))
mem_usage_test_ori =(mem_usage(test))
mem_usage_test_off_ori =(mem_usage(test_off))

mprint(len(train),'len(train)')
mprint(len(valid),'len(valid)')
mprint(len(test),'len(test)')
mprint(len(test_off),'len(test_off)')

####    开始ONEHOT 编码和稀疏向量化

train_x=train[['creativeSize']]
valid_x=valid[['creativeSize']]
test_x=test[['creativeSize']]
test_off_x=test_off[['creativeSize']]
mprint('onehot_trans begin')
len_one_hot_feature=len(one_hot_feature)
for feature in one_hot_feature:
    enc = OneHotEncoder()
    enc.fit(data[feature].values.reshape(-1, 1))
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
mprint(train_x.shape,'train_x.shape')
mprint(valid_x.shape,'valid_x.shape')
mprint(test_x.shape,'test_x.shape')
mprint(test_off_x.shape,'test_off_x.shape')

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





if params_flag ==False:
    ### 数据转换
    mprint('数据转换')
    lgb_train = lgb.Dataset(train_x, train_y, free_raw_data=False)
    lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train,free_raw_data=False)
    
    
    ### 设置初始参数--不含交叉验证参数
    mprint('设置参数')
    # params = {
    #           'boosting_type': 'gbdt',
    #           'objective': 'binary',
    #           'metric': 'auc',
    # #          'max_depth':-1,
    # #          'min_data_in_leaf':20,
    # #          'feature_fraction':1.0,
    #           }
    
    params ={'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['auc'], 'num_leaves': 85, 'max_depth': 7, 
    'verbose': 1, 'max_bin': 255, 'min_data_in_leaf': 100, 'feature_fraction': 0.6, 'bagging_fraction': 0.8, 'bagging_freq': 40}
    ### 交叉验证(调参)
    mprint('交叉验证')
    min_merror = float('-Inf')
    ##初始化
    best_params ={'max_depth': -1, 'min_split_gain': 0, 'verbose': 1, 'lambda_l2': 0, 'num_leaves': 31,
                  'feature_fraction': 1.0 ,'objective': 'binary', 'max_bin': 255,'boosting_type': 'gbdt', 'min_data_in_leaf': 100, 
                  'bagging_fraction': 1.0, 'bagging_freq': 0, 'lambda_l1': 0, 'metric': ['auc']}
    
    # 准确率
    mprint("调参1：提高准确率")
    for num_leaves in range(25,130,5):
        for max_depth in range(5,8,1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
    
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=2018,
                                nfold=3,
                                metrics=['auc'],
                                early_stopping_rounds=5,
                                verbose_eval=True,
                                )
            mean_auc_value = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).argmax()

            mprint('lgb.cv runone!%s, run mean_auc_value:%s ,boost_rounds:%s,num_leaves:%s ,max_depth:%s'%(str('step1 Accuracy'),str(mean_auc_value),str(boost_rounds),str(num_leaves),str(max_depth)))
            mprint('the best auc right now is %s'%(min_merror))
            if mean_auc_value > min_merror:
                min_merror = mean_auc_value
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
                mprint(mean_auc_value,'better auct is get :mean_auc_result_step1')  
    params['num_leaves'] = best_params['num_leaves']
    params['max_depth'] = best_params['max_depth']
    
    mprint(params,'best_params_step1')
    mprint(min_merror,'final auc of step1!')
    
    
    # 过拟合
    mprint("调参2：降低过拟合")
    for max_bin in range(1,255,5):
        for min_data_in_leaf in range(100,5000,500):       
                params['max_bin'] = max_bin
                params['min_data_in_leaf'] = min_data_in_leaf
                
                cv_results = lgb.cv(
                                    params,
                                    lgb_train,
                                    seed=42,
                                    nfold=3,
                                    metrics=['auc'],
                                    early_stopping_rounds=5,
                                    verbose_eval=True,
                                    )
                        
                mean_auc_value = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
                mprint('lgb.cv runone!%s, run mean_auc_value:%s ,boost_rounds:%s,max_bin:%s ,min_data_in_leaf:%s'%(str('step2 eroverfit'),str(mean_auc_value),str(boost_rounds),str(max_bin),str(min_data_in_leaf)))
                mprint('the best auc right now is %s'%(min_merror))
                if mean_auc_value > min_merror:
                    min_merror = mean_auc_value
                    best_params['max_bin']= max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
                    mprint(mean_auc_value,'better auct is get :mean_auc_result_step2')  
    params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    params['max_bin'] = best_params['max_bin']
    mprint(params,'best_params_step2')
    mprint(min_merror,'final auc of step2!')
    
    
    mprint("调参3：降低过拟合")
    for feature_fraction in [0.4,0.5,0.6,0.7,0.8]:
        for bagging_fraction in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            for bagging_freq in range(0,50,5):          
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
                                    verbose_eval=True,
                                    )
                        
                mean_auc_value = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
                mprint('lgb.cv run one!%s,run mean_auc_value:%s ,boost_rounds:%s,feature_fraction:%s ,bagging_fraction:%s,bagging_freq:%s'%(str('step3 loweroverfit'),str(mean_auc_value),str(boost_rounds),str(feature_fraction),str(bagging_fraction),str(bagging_freq)))
                mprint('the best auc right now is %s'%(min_merror))

                if mean_auc_value > min_merror:
                    min_merror = mean_auc_value
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq
                    mprint(mean_auc_value,'better auct is get :mean_auc_result_step3')  
    params['feature_fraction'] = best_params['feature_fraction']
    params['bagging_fraction'] = best_params['bagging_fraction']
    params['bagging_freq'] = best_params['bagging_freq']
    mprint(params,'best_params_step3')
    mprint(min_merror,'final auc of step3!')
    

    mprint("调参4：降低过拟合")
    for lambda_l1 in [0.0,0.2,0.2,0.4,0.6,0.8,1.0]:
        for lambda_l2 in [0,0,0.2,0.4,0.6,0.8,1.0]:
            for min_split_gain in [0.0,0.2,0.4,0.6,0.8,1.0]:
                params['lambda_l1'] = lambda_l1
                params['lambda_l2'] = lambda_l2
                params['min_split_gain'] = min_split_gain

                
                cv_results = lgb.cv(
                                    params,
                                    lgb_train,
                                    seed=42,
                                    nfold=3,
                                    metrics=['auc'],
                                    early_stopping_rounds=10,
                                    verbose_eval=True,
                                    )
                        
                mean_auc_value = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
                mprint('lgb.cv run one!%s,run mean_auc_value:%s ,boost_rounds:%s,lambda_l1:%s ,lambda_l2:%s,min_split_gain:%s'%(str('step4 loweroverfit'),str(mean_auc_value),str(boost_rounds),str(lambda_l1),str(lambda_l2),str(min_split_gain)))
                if mean_auc_value > min_merror:
                    min_merror = mean_auc_value
                    best_params['lambda_l1'] = lambda_l1
                    best_params['lambda_l2'] = lambda_l2
                    best_params['min_split_gain'] = min_split_gain
                    mprint(mean_auc_value,'better auct is get :mean_auc_result_step4')  
                    mprint('the best auc right now is %s'%(min_merror))
                    # mprint('the best params right now is %s'%(str(params)))
    params['lambda_l1'] = best_params['lambda_l1']
    params['lambda_l2'] = best_params['lambda_l2']
    params['min_split_gain'] = best_params['min_split_gain']
    
    mprint(params,'best params')
    mprint(min_merror,'final auc of cross_valid test!')
    gc.collect()
    try:

        with open(path_bestparams,"w") as f:
                f.write(str(params))
        mprint('best_paramss is wirted into %s'%(path_bestparams))

    except :
        mprint('write best_parmas error')
else:
##params
    params ={'max_depth': 6, 'min_split_gain': 0.1, 'verbose': 1, 'lambda_l2': 0.1, 'num_leaves': 40, 'feature_fraction': 0.1, 'objective': 'binary', 'max_bin': 1, 'boosting_type': 'gbdt', 'min_data_in_leaf': 30, 'bagging_fraction': 0.8, 'bagging_freq': 10, 'lambda_l1': 0.1, 'metric': ['auc']}
    with open(path_bestparams,"w") as f:
            f.write(str(params))
    mprint('best_paramss is wirted into %s'%(path_bestparams))


def my_LGB_test(train_x,train_y,test_x,test_y):
#    from multiprocessing import cpu_count
    mprint("LGB test")
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
    mprint("LGB predict")
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
    try:
        date = datetime.now().strftime('%Y%m%d-%H')
        remote_path = str(date)+'_submission.csv'
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



#begin to run 

# model_test = my_LGB_test(train_x,train_y,test_off_x,test_off_y)
# model_predict = my_LGB_predict(train_x,train_y,train_x,train_y,test_x,res)
mprint('model_params is fitted!')
del train_x
del train_y
del valid_x
del valid_y
del test_off_y
del test_off_x
del test_x 
gc.collect()
mprint('del the key variable and free memory!')
mprint('run the final py') 
import ad_res
