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
    path_data_dtypes = '/root/workspace/data/data_dtypes.txt'
    path_data_hdf5='/root/workspace/data/data_prepared_2.hdf5'
    path_data_csv='/root/workspace/data/data_prepared.csv'
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
    path_data_dtypes = 'E:/MLfile/preliminary_contest_data/data/data_dtypes.txt'
    path_data_hdf5='E:/MLfile/preliminary_contest_data/data/data_prepared_2.hdf5'
    path_data_csv='E:/MLfile/preliminary_contest_data/data/data_prepared.csv'

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
    chunk =500000
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
    mprint(dtype,'feature %s dtype'%(col))
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
    mprint(dtype,'feature %s dtype'%(col))
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
readnum = 500000

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

## DATA DTYPES SAVES

dtypes = data.dtypes
dtypes_col = dtypes.index
dtypes_type = [i.name for i in dtypes.values]

column_types = dict(zip(dtypes_col, dtypes_type))

with open(path_data_dtypes,"w") as f:
        f.write(str(column_types))

#try:
#    
#    h5_file = h5py.File(path_data_hdf5,'w')
#    for col in data.columns:
#        dtype = data[col].dtypes
#        mprint(dtype)
#        dset =h5_file.create_dataset(name =col,data = data[col],dtype= dtype)
#        mprint(' h5py  create_dataset %s sucessed'%(col))
#        mprint('h5_file sucessedd')
#except:
#    mprint('h5_file failed')

try:
    data.to_csv(path_data_csv)
    mprint('data_to_csv finished!')
except:
    mprint('data_to_csv failed!')

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
mem_usage_data_ori =(mem_usage(data))
mem_usage_train_ori =(mem_usage(train))

mem_usage_valid_ori =(mem_usage(valid))

mem_usage_test_ori =(mem_usage(test))
mem_usage_test_off_ori =(mem_usage(test_off))


####    开始ONEHOT 编码和稀疏向量化

train_x=train[['creativeSize']]
valid_x=valid[['creativeSize']]
test_x=test[['creativeSize']]
test_off_x=test_off[['creativeSize']]
mprint('onehot_trans begin')
for feature in one_hot_feature:
    enc = OneHotEncoder()
    enc.fit(data[feature].values.reshape(-1, 1))
    mprint(enc.n_values_,'feature:%s enc.n_values_'%(feature))
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
    mprint (mem_usage(data),'mem_usage(data) after %s'%(feature))
    mprint (mem_usage(train),'mem_usage(train) after %s'%(feature))
    mprint (mem_usage(valid),'mem_usage(valid) after %s'%(feature))
    mprint (mem_usage(test),'mem_usage(test) after %s'%(feature))
    mprint (mem_usage(test_off),'mem_usage(test_off) after %s'%(feature))

    mprint('feature:%s one-hot finished!'%(feature))

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

    mprint (mem_usage(data),'mem_usage(data) after %s'%(feature))
    mprint (mem_usage(train),'mem_usage(train) after %s'%(feature))
    mprint (mem_usage(valid),'mem_usage(valid) after %s'%(feature))
    mprint (mem_usage(test),'mem_usage(test) after %s'%(feature))
    mprint (mem_usage(test_off),'mem_usage(test_off) after %s'%(feature))

    mprint('feature:%s CountVectorizer finished!'%(feature))

mprint('countvec_trans prepared !')

mprint((mem_usage_data_ori),'mem_usage(data) ori ')

mprint((mem_usage_train_ori),'mem_usage(train) ori ')
mprint ((mem_usage_train_ori),'mem_usage(valid) ori ')
mprint ((mem_usage_test_ori),'mem_usage(test) ori ')
mprint ((mem_usage_test_off_ori),'mem_usage(test_off) ori ')


## 尝试KMEANS聚类
#from sklearn.cluster import MiniBatchKMeans
#classes_train =[]
#mbk1 = MiniBatchKMeans(n_clusters=500, init=’k-means++’, max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None,tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
#timespent('begin_MiniBatchKMeans')
#mbk.fit(train_x)
#timespent('begin_MiniBatchKMeans')
#classes_train = np.append(classes_1, mbk.labels_)

## 尝试KMEANS聚类
'''
def onehot_n_countvec_trans():
    sample=[train,valid,test,test_off]
    sample_name=['train','valud','test','test_off']
    data = data
    len_samplelist = len(sample)
    sample_x=[]
    for i in range(len_samplelist):
        print (type(sample[i]))
        #空数组 不能越界
        sample_x[i].append(sample[i][['creativeSize']])

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
    returnlist=[]
    for i in range(len_samplelist):
        returnlist.append(sample_x[i])
    gc.collect()
    return returnlist
'''


#model=LGB_predict(train_x,train_y,test_x,res)

if params_flag ==False:
    ### 数据转换
    mprint('数据转换')
    lgb_train = lgb.Dataset(train_x, train_y, free_raw_data=False)
    lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train,free_raw_data=False)
    
    
    ### 设置初始参数--不含交叉验证参数
    mprint('设置参数')
    params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric': 'auc',
    #          'max_depth':-1,
    #          'min_data_in_leaf':20,
    #          'feature_fraction':1.0,
              }
    
    ### 交叉验证(调参)
    mprint('交叉验证')
    min_merror = float('-Inf')
    ##初始化
    best_params ={'max_depth': -1, 'min_split_gain': 0, 'verbose': 1, 'lambda_l2': 0, 'num_leaves': 31,
                  'feature_fraction': 1.0 ,'objective': 'binary', 'max_bin': 255,'boosting_type': 'gbdt', 'min_data_in_leaf': 100, 
                  'bagging_fraction': 1.0, 'bagging_freq': 0, 'lambda_l1': 0, 'metric': ['auc']}

    # 准确率
    mprint("调参1：提高准确率")
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
                
            mean_auc_value = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
                
            if mean_auc_value > min_merror:
                min_merror = mean_auc_value
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
                mprint(mean_auc_value,'mean_auc_result_step1')  
    params['num_leaves'] = best_params['num_leaves']
    params['max_depth'] = best_params['max_depth']
    #'''
    mprint(params,'best_params_step1')
    # 过拟合
    mprint("调参2：降低过拟合")
    for max_bin in range(1,255,5):
        for min_data_in_leaf in range(10,200,5):       
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
                        
                mean_auc_value = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
    
                if mean_auc_value > min_merror:
                    min_merror = mean_auc_value
                    best_params['max_bin']= max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
                    mprint(mean_auc_value,'mean_auc_result_step2')  
    params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    params['max_bin'] = best_params['max_bin']
    mprint(params,'best_params_step2')
    
    mprint("调参3：降低过拟合")
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
                        
                mean_auc_value = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
    
                if mean_auc_value > min_merror:
                    min_merror = mean_auc_value
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq
                    mprint(mean_auc_value,'mean_auc_result_step3')              
    params['feature_fraction'] = best_params['feature_fraction']
    params['bagging_fraction'] = best_params['bagging_fraction']
    params['bagging_freq'] = best_params['bagging_freq']
    mprint(params,'best_params_step3')
    
    mprint("调参4：降低过拟合")
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
                        
                mean_auc_value = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
    
                if mean_auc_value > min_merror:
                    min_merror = mean_auc_value
                    best_params['lambda_l1'] = lambda_l1
                    best_params['lambda_l2'] = lambda_l2
                    best_params['min_split_gain'] = min_split_gain
                    mprint(mean_auc_value,'mean_auc_result_step4')  
    params['lambda_l1'] = best_params['lambda_l1']
    params['lambda_l2'] = best_params['lambda_l2']
    params['min_split_gain'] = best_params['min_split_gain']
    
    mprint(params,'best params')
    gc.collect()
else:
##params
    params ={'max_depth': 6, 'min_split_gain': 0.1, 'verbose': 1, 'lambda_l2': 0.1, 'num_leaves': 40, 'feature_fraction': 0.1, 'objective': 'binary', 'max_bin': 1, 'boosting_type': 'gbdt', 'min_data_in_leaf': 30, 'bagging_fraction': 0.8, 'bagging_freq': 10, 'lambda_l1': 0.1, 'metric': ['auc']}

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



model_test = my_LGB_test(train_x,train_y,test_off_x,test_off_y)
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

mprint('PROGRAM IS ENDDING!')    


mail('ad is all done!')

